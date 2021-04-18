
import re
import sys
import ntpath

from string import punctuation

import fitz
import numpy as np
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from rapidfuzz import fuzz

# MuPDF is a bit over-verbosed on warnings.
# We only want to handle exceptions
fitz.TOOLS.mupdf_display_errors(False)
# For determenistic language prediction
DetectorFactory.seed = 0

try:
    import ocrmypdf
    OCR_AVAILABLE = True  # Use OCR or skip files without text layer
except ImportError:
    print('OCR not available, please install OCRMyPDF (see README)')
    OCR_AVAILABLE = False

KEYLINES_FILENAME = 'Strings to identify SDS all languages v5.xlsx'
LINES_FOR_OCR = 30  # if less than this number of overall lines is found in text - OCR
MULTI_SDS_MIN_PAGE_COUNT = 25  # Minimum page count to treat PDF as multi SDS
PUNCTUATION = set(punctuation)  # basic punctuation special symbols
PUNCTUATION.remove('.')  # Often goes as alignment in agendas
# Stop condition Regexp: we are trying to find first Section 3 subsection, like:
# 3.1 substances, 3.2 mixtures, 3.2 chemical characterisation: mixtures and so on
# Regexp has to be strong to exclude false positive stops (tested on 3800 pdfs)
STOP_PATTERN = re.compile(r'3\.[12][\.\s]+[a-z]{5,}')


class InvalidTextError(Exception):
    '''Exception for empty/wrong language text layer document'''
    pass


def get_language(text):
    '''
    Determine language text.

    Also determine if record is basically a text:
    a lot of trash text layers are consist of punctuation mainly.
    '''
    # Punctuation ratio for given trash text
    if not len(text):
        return 'unknown'

    punct_ratio = len([c for c in text if c in PUNCTUATION]) / len(text)
    if punct_ratio > 0.5:
        return 'unknown'

    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return 'unknown'


def get_reference_data(file):
    '''
    Reference lines and Section 3 candidates for each available language.

    File notation for beginning of line:
    $ - helper_line. We use this line only for order metric, not distance metric.
        Also these lines are not relevant for short documents.
        Good examples are: UNCOMPLETE lines such as address, email, and so on

    Returns a tuple with:

    1. All keylines with needed metadata for each language
    2. Section 3 header candidates for each language
    '''
    def parse_line(line):
        '''Parsing line semantics'''

        helper_line = False
        line = line.lower()

        if line.startswith('$'):
            # only relevant for order score
            line = line.lstrip('$').strip()
            helper_line = True

        return {
            'line': line,
            'helper_line': helper_line
        }

    lines_df = pd.read_excel(file, sheet_name='string_list')
    lines = lines_df.groupby('lang_short').line.apply(lambda lines: [parse_line(ln) for ln in lines])

    section3_candidates_df = pd.read_excel(file, sheet_name='section3_candidates')
    section3_candidates = section3_candidates_df.groupby('lang_short').apply(
        lambda x: {ln: s for ln, s in zip(x.line, x.min_similarity)}
    )

    layout_markers_df = pd.read_excel(file, sheet_name='layout_markers')
    layout_markers = layout_markers_df.set_index('lang_short').to_dict(orient="index")
    return lines.to_dict(), section3_candidates.to_dict(), layout_markers


def get_document_data(file, min_length=5):
    '''
    Basic page-to-page text extraction,
    cleansing, and line splitting. Also returns all
    the document metadata needed
    '''
    try:
        meta = {}
        with fitz.open(file) as doc:
            meta['page_count'] = len(doc)
            lines = []

            for page in doc:
                page_text = page.get_text('text')

                # Cleansing
                page_text = page_text.lower()
                page_text = re.sub(r'[\*�]', '', page_text)  # artefacts of bad text layer
                for line_text in page_text.split('\n'):
                    line_text = line_text.replace('\t', ' ')
                    line_text = re.sub(r'\s+', ' ', line_text)

                    if len(line_text) < min_length:
                        # too short lines causes noise in similarity
                        continue

                    lines.append(line_text.strip())
    except RuntimeError as e:
        # Unfortunately there are no specific exceptions on corrupted file,
        # so we reraise with filename and MuPDF error text
        raise RuntimeError(f'Corrupted PDF: {file}, MuPDF says: {str(e)}')

    # All lines is needeed to scan for multi SDS
    meta['all_lines'] = lines
    relevant_lines = lines

    # Try to limit the search scope with section 3
    for i, line in enumerate(lines):
        # Specific stop condidion: if we reach section 3 subsections
        if STOP_PATTERN.match(line):
            relevant_lines = lines[:i]
            break

    return relevant_lines, meta


def get_candidate_lines(keylines, lines, meta):
    """
    For each key line get the candidate
    line from file lines with similarity metrics
    and line position
    """
    results = []

    # Heuristic 1: limit the search scope with the best candidate
    # for the LAST reference line
    last_line_candidates = []

    # Find the best candidate
    for position, line in enumerate(lines):
        similarity = fuzz.ratio(keylines[-1]['line'], line)
        last_line_candidates.append((similarity, position, line))

    cand_score, cand_pos, cand_line = max(last_line_candidates)

    # if it is not a random match (at least 55%)
    # and it cointains number "3", limit the search scope
    # (because in rare cases sections 2 and 3 are swapped)
    if (cand_score > 55 and '3' in cand_line):
        search_scope = lines[:cand_pos + 1]  # +1 because the line itself is very valuable for matching
    else:
        search_scope = lines

    # Heuristic 2: for very short documents (2-5 pages)
    # there often will be only sections with brief info
    # no adresses and detailed descriptions,
    # so we dont take some lines into account
    if 1 < meta['page_count'] < 5:
        keyline_scope = [kl for kl in keylines if not kl['helper_line']]
    else:
        keyline_scope = keylines

    # Now pick the best candidate for each reference line
    for keyline in keyline_scope:
        candidates = []

        for position, line in enumerate(search_scope):
            similarity = fuzz.ratio(keyline['line'], line)
            candidates.append({
                'line': line,
                'similarity': similarity,
                'position': position
            })

        best_match = max(candidates, key=lambda c: c['similarity'])
        results.append({'keyline': keyline, **best_match})

    return results

def get_pdf_strings(file, type, start, stop, min_length=5):
    stoppattern = ""
    startptrn =""
    sec2ptrn = r'(((2)?(\s.\s)?(\:)?(\.)?(\s+)?Hazards\sidentification)|(TRANSPORTATION\sINFORMATION))'
    if type == 1:
        stoppattern = r'(1\.2(\.)?)?(\s+)?(Relevant\sidentified(.+)(\s)|Product\sUses|^SUPPLIER)'
    elif type == 2 or type == 3:
        startptrn = r'((Details\sof\sthe\ssupplier(\s+))|(NAME\sOF\sMANUFACTURER/SUPPLIER))'
        stoppattern = r'(1\.4)(\.)?(\s+)?(Emergency\stelephone\snumber)'

    contents = ""
    try:
        with fitz.open(file) as doc:
            if stop is not True:
                for page in doc:
                    page_blocks = page.get_text('blocks')
                    page_blocks.sort(key=lambda block: block[1])
                    # Extracting text:
                    if stop is not True:
                        for block in page_blocks:
                            block_type = block[6]
                            if block_type == 1:  # ignore image blocks
                                continue
                            block_text = block[4]

                            section1_3 = re.search(startptrn, block_text, re.IGNORECASE)
                            if section1_3 is not None:
                                start = True
                            if type == 1:
                                contents += block_text
                            elif type == 2 or type == 3:
                                if start == True:
                                    contents += block_text
                            section2 = re.search(sec2ptrn, block_text, re.IGNORECASE)
                            section1_2 = re.search(stoppattern,block_text, re.IGNORECASE|re.MULTILINE)
                            if section2 is not None or section1_2 is not None:
                                stop = True
                                break

    except RuntimeError as e:
        # Unfortunately there are no specific exceptions on corrupted file,
        # so we reraise with filename and MuPDF error text
        raise RuntimeError(f'Corrupted PDF: {file}, MuPDF says: {str(e)}')

    if start is False:
        contents = get_pdf_strings(file, type, True, False)

    return contents

def get_productname(contents, pdf_file, type):
    result = ""
    patterns = [r'^(\s+)?Name\sof\sthe\ssubstance(\s+)?(.+)(\s+)?',
                r'(1\.1)?(ꞏ|-||∙|·|•|·|\*)?(\s+)?(Product\sidentifier)(\(s\))?(\s+)?(.+)(\s+(·|•)\s+Trade\sname)(\s+)?(.+)(\s+)?',
                r'(1\.1)?(ꞏ|-||∙|·|•|·|\*)?(\s+)?(Product\sidentifier)(\(s\))?(\s+)?(.+)(\s+(·|•)\s+Trade\sname)',
                r'^Product\sTrade\-name(\:)?(\s+)?(.+)(\s+)?',
                r'^Product Name/Nom commercial du produit(\s+)?(.+)(\s+)?',
                r'^(\s+)?(.)?(\s+)?(Chemical\s/\s)?Trade(\scode\sand)?(\s)?name(\sand\sproduct\snumber)?(s)?(\sand)?(\(s\))?(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'^(1\.1)?(ꞏ|-|∙|·|·|•|\*)?(ꞏ)?(\s+)?(Product(\s+)?Name/\s)?(Trade(\s)?name)(\s\(As\sLabeled\))?(\s/\sArticle\-No)?(\sor)?(\s+)?(designation\sof\sthe\smixture)?(/\sSubstance\sname)?(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'^(\s+)?(1\.1)?(ꞏ|-|∙|·|·|•|\*|�+)?(\s+)?(Material/)?Trade(\s+)?name(\s+)?(or|/)(\s+)?designation(\s+)?(of\sthe\smixture)?(\:)?(\s+)?(.+)(\s)',
                r'^(1\.1)?(ꞏ|-|∙|·|·|•|\*|�+)?(\s+)?(Material/)?Trade(\s+)?name(\s+)?(\(as(\s+)?labeled\))?(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'^(\s+)?Identification\son\sthe\slabel/Trade(\s+)name(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'(\s+)?(Product(\s+)?(name|description)(\.)?(\(s\))?(\s\(product\sidentifiers\))?(?!d)(s)?)(\s+)?(\:)?(\s+)?(\^\s)?(.+)(\s+)?',
                r'(\s+)?(Product(\s+)?(name|description)(^d)?(s)?)(\sand/or\scode)?(\s+)?(\:)?(\s+)?(\.)?(\s+)?(\^)?(\s+)?(.+)(\s)',
                r'(\s+)?Product(\sTrade\s|\s)?name(\s+)?(\:)(\s+)?(.+)(\s+)?',
                r'(\s+)?(Product|Business)\sname(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Name\sof\sthe\ssubstance(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'^Substance\sName(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'^Color\sName(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'^(\s+)?Identification\sof\sthe(\ssubstance)?(\sor)?\spreparation(?!\sAND\sTHE\sCOMPANY/UNDERTAKING)(\s+)?(\:)?(\s+)?(Name)?(\:)?(\s+)?(.+)(\s+)?',
                r'^Identification on the label(\s)?/(\s)?Trade name(\s+)?(label designation/Name of product)?(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'Identification\sof\sthe(\s+)?substance(\s+)?(\sor\s)?((/)?preparation|(/)?mixture)?((\s+)?and(\s+)?COMPANY/UNDERTAKING(\s+)?)?(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'Identification\sof\sthe\ssubstance/mixture\sand(\sof)?\sthe\scompany/undertaking(\s+)?(.+)(\s+)?',
                r'Identification\sof\sthe\ssubstance\sor(\s+)?preparation(\s+)?(.+)(\s+)?',
                r'^Designation\s/\sCommercial\sname(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'^name\s\(Synonyms\)(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'name\sof\s(the\s)?(product|Substance)(s)?(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'^(Commercial\s)?name(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'(Brand(\s)?name|Full(\s)?Name)(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Substance\sname/trade\sname(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'(1\.1)?(ꞏ|-|∙|·|•|\*)?(\s+)?(Product\sidentifier)(/TRADE/MATERIAL\sNAME)?(\(s\))?(\s+)?(Material(\sname))?(\s+)?(\:|•)?(\s+)?(.+)(\s)',
                r'(1\.1)?(ꞏ|-|∙|·|•|\*)?(\s+)?(Product\sidentifier)(/TRADE/MATERIAL\sNAME)?(\(s\))?(\s+)?(Product\sidentification.+)(\s+)?(Name)(:|•)?(\s)?(.+)(\s)',
                r'(1\.1)?(ꞏ|-|∙|·|•|\*)?(\s+)?(Product\sidentifier)(/TRADE/MATERIAL\sNAME)?(\(s\))?(\s+)(Product\sidentifier)(\:|•)?(\s+)(.+)(\s)',
                r'(1\.1)?(ꞏ|-|∙|·|•|\*)?(\s+)?Product\sidentifier(/TRADE/MATERIAL\sNAME)?(\(s\))?(\s+)?Trade\sname/designation(\s+)?(\:|•)?(\s+)?(.+)(\s)',
                r'(1\.1)?(ꞏ|-|∙|·|•|\*)?(\s+)?(Product\sidentifier)(/TRADE/MATERIAL\sNAME)?(\(s\))?(\:)?(\s+)?(Name\s\(Synonyms\)\:|•)(\s+)?(.+)(\s)',
                r'(1\.1)?(ꞏ|-|∙|·|•|\*)?(\s+)?(Product\sidentifier)(/TRADE/MATERIAL\sNAME)?(\(s\))?(\:)?(\s+)(Products\sname)(\s+)(\^\s)?(.+)(\s)',
                r'(1\.1)?(ꞏ|-|∙|·|•|\*)?(\s+)?(Product\sidentifier)(/TRADE/MATERIAL\sNAME)?(\(s\))?(\:)?(\s+)?(Preparation\sCommercial\sName)(\:)?(\s+)(.+)(\s)',
                r'(1\.1)?(ꞏ|-|∙|·|•|\*)?(\s+)?(Product\sidentifier)(\.)?(/TRADE/MATERIAL\sNAME)?(\(s\))?(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'Identification(\s+)?of(\s+)?the(\s+)?product(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Identification(\s+)?of(\s+)?the(\s+)?mixture(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'(Identification\sof\sthe\ssubstance/mixture\sand\sof\sthe\scompany/undertaking)(\s+)(.+)(\s)',
                r'IDENTIFICATION\sOF\sTHE\sSUBSTANCE(\s)?/(\s)?PREPARATION\sAND(\sOF\sTHE(\s+)?COMPANY)?(\s)?(/)?(\s)?(UNDERTAKING)?(\s+)?(\:)?(\s+)?(.+)(\s)',
                r'IDENTIFICATION\sOF\sSUBSTANCE\s\&\sCOMPANY\sIDENTIFICATION(\s+)?(Product\:)?(\s+)?(.+)(\s+)?',
                r'(Identification\sof\sthe\ssubstance\sor\smixture)(\s+)(.+)(\s)',
                r'(Identification\sof\s(the\s)?substance\s(or\s|/\s)?preparation)(\sand\sof\sthe\scompany)?(\s+)?(Product)?(\:)?(\s+)?(.+)(\s)',
                # r'^(\s+)?Manufacturer(/Supplier)?(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'^(\s+)?Product\sand\scompany\sidentification(\s+)?(1)?(\s+)?(.)?(\s+)(\:)?(\s+)?(.+)(\s+)?',
                r'^Chemical\sproduct\sidentification\sSample(\s+)?Description(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'IDENTIFICATION\sOF\sTHE\sPRODUCT(?! and of the company)(\s+)?(.+)(\s+)?',
                r'Product\s(identified|identification)(\:)?(\s+)?(.+)(\s)',
                r'PRODUCTIDENTIFICATIE\:(\s+)?(.+)(\s)',
                r'(\s+)?Identification\sof\sthe\ssubstance(/mixture|\s/\sPREPARATION\sOF\sTHE\sCOMPANY/UNDERTAKING)?(\s+)?(.+)(\s+)?',
                r'^IDENTITY \(As Used on Label and List\)(\s+)?(.+)(\s+)?',
                r'Type\sof\sproduct(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'^Product\sidentification(\s+)?(.+)(\s+)?',
                r'^(· )?Product(\s+)?(\:)(\s+)?(.+)(\s+)?',
                r'^Common\sname(\s+)?(\:)(\s+)?(.+)(\s+)?',
                r'^Product Identity(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'^PREPARATION\sNAME(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'^MSDS\sNAME(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Product\sdesignation(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Product\scommercial\sname(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Material\sIdentity(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Products:(\s+)?(.+)(\s+)?',
                r'Nombre\scomercial\s/(\s+)?(.+)(\s+)?',
                r'NAME OF THE COMPONENT(\s+)?(.+)(\s+)?',
                r'Product Name\:(\s+)?(.+)(\s+)?Product Code\:.+\s+',
                r'Product\sGroup\:(\s+)?(.+)(\s+)?',
                r'PREPARATION NAME\:(\s+)?(.+)(\s+)?Product Code\:.+\s+',
                r'SDS\sNr(\s+)?(\:)?(.+)(\s+)?',
                r'(\s+)?Product\sName(\s+)?(\:)?(\s+)?(.+)(\s+)?(.+)(\s+)?',
                r'Description\:(\s+)?(.+)(\s+)?',
                r'Name of the Mixture\:(\s+)?(.+)(\s+)?',
                r'(IDENTIFICATION OF )?SUBSTANCE\:(\s+)?(.+)(\s+)?',
                r'Product indentifier(\s+)?(.+)(\s+)?',
                r'Material Name(\s+)?\:(\s+)?(.+)(\s+)?',
                r'Name\:(\s+)?(.+)(\s+)?',
                r'Identifikátor výrobku(\s+)?(.+)(\s+)?',
                r'Material\:(\s+)?(.+)(\s+)?',
                r'GHS Product\sidentifier(s)?(\s+)?(.+)(\s+)?',
                r'Product\sidentifier(s)?(\s+)?(.+)(\s+)?',
                r'Unique Formula Identifier\:(\s+)?(.+)(\s+)?'
                r'Chemical\s(description|name)(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Information on the product\:(\s+)?(.+)(\s+)?',
                r'Identification of the(\s+)?substance/preparation(\s+)?(.+)(\s+)?',
                r'Identification of the(\s+)?substance\sor(\s+)?preparation(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Product description(\s+)?(\:)?(\s+)(.+)(\s+)',
                r'Identificatore del prodotto(\s+)(.+)(\s+)',
                r'Product\:(\s+)?(.+)(\s+)?',
                r'Substances / Trade name / designation\:\s(.+)(\s+)?'
                r'Bezeichnung\sder\sZubereitung(\s+)?(.+)(\s+)?',
                r'Identification\sof\sthe\spreparation(\s+)?(.+)(\s+)?',
                r'^Identification\sof\sthe\ssubstance(\s+)?(.+)(\s+)?',
                r'^Identification\sof\sthe\ssubstance/mixture(\s+)?(.+)(\s+)?',
                r'^name\sof\sthe\ssubstance(\s+)?(.+)(\s+)?',
                ]

    found = False
    for f in patterns:
        pname = re.search(f, contents, re.IGNORECASE | re.MULTILINE)
        if pname is not None:
            if pname.group(len(pname.groups()) - 1).lower().find("Product code") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product form") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("code") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("1.") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("cas-no.") == False \
                    and pname.group(len(pname.groups()) - 1).lower().find("ufi") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().startswith(".") == False \
                    and pname.group(len(pname.groups()) - 1).lower().find("product identity") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("for professional use only") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product description") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product indentifier") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("description") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("designation") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("of the mixture") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("application") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("ec number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("(as labeled): ") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("(as labeled): ") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("common name:") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("trade name:") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product no.") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("(see page 1)") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("/ preparation") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("/ substance name") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("of the company/undertaking") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("identification / trade name:") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("/business name: ") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("and function of the responsible") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("cas no. ") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("composition and use") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("and supplier") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("and/or code") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("are covered by a limit") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("art.-no.") == False \
                    and pname.group(len(pname.groups()) - 1).lower().find("article number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("further trade name") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("and of the") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("manufacturer/supplier") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("/supplier") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("material uses") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("not known") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("msds code") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("reach substance name") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product use") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("and of the") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("address") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("reach registration number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("registration number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("part no.") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("and synonyms") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product identifier") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("identification of the product") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("identification of the substance") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product:") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product part number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("reference number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product no:") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product identity") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("index number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("of product") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("trade name") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("commercial name") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product information") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("s:") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("name:") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("company undertaking") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("& synonyms") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("part no.") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("product identifier") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("company/undertaking") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("/ mixture") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("/mixture") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("or mixture") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("manufacturer") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("information about the company") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("information on the product") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("information on the product") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("distributor:") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("cas number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("date of issue") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("sds no.") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("d below.") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("name of substance") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("identification of the") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("chemical name") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("cas number") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("identification") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("material:") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("name of the substance") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("identity (as used on label and list)") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().find("no further relevant information available") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("d above.") == False \
                    and pname.group(len(pname.groups()) - 1).lower().find("identified use") == -1 \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("n/a") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("liquid.") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("telephone:") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("and of") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("part number:") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("reach registration name") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("substance name") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("product	name") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("product name") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("/  mixture") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("and manufacturer") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("synonyms") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("for research use only") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("of the component") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("(chemical name):") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("chemical description") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("pure product") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("and the") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("and the company") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("and of the company/undertaking") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("preparation name") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("sds nr") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("revision date") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("of the mixture") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("& Of the vompany/undertaking") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("and company") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("and identification of the") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("see product identifier") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("Product No") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("material name") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("company/undertaking") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("and of the company/undertaking") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("this is a personal care or") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("product identification") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("none known") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("technical product") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("not available") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("unique formula identifier") == False \
                    and pname.group(len(pname.groups()) - 1).lower().startswith("of the") == False \
                    and pname.group(len(pname.groups()) - 1) != "：" \
                    and pname.group(len(pname.groups()) - 1) != "" \
                    and pname.group(len(pname.groups()) - 1) != "s" \
                    and pname.group(len(pname.groups()) - 1) != "-" \
                    and pname.group(len(pname.groups()) - 1) != ":" \
                    and pname.group(len(pname.groups()) - 1) != ": " \
                    and pname.group(len(pname.groups()) - 1) != ")" \
                    and pname.group(len(pname.groups()) - 1) != " " \
                    and pname.group(len(pname.groups()) - 1).lower() != "and" \
                    and pname.group(len(pname.groups()) - 1).lower() != "of " \
                    and pname.group(len(pname.groups()) - 1).lower().find("name (synonyms)") == -1:
                result = pname.group(len(pname.groups()) - 1).split('(Contains')[0].replace(': ', '').replace(':','').replace('●', '').rstrip()
                if len(result) <= 2:
                    result = pname.group(len(pname.groups()) - 1).split('(Contains')[0].replace(': ', '').replace(':','').replace('●', '').rstrip()
                found = True
                result = re.sub(r"^\:(\s+)?", "", result).rstrip()
                break
    if found is not True:
        result = "Not found"
    return result


def get_companyname(contents, pdf_file, type):
    result = ""
    patterns = [r'^(\s+)?(\:)?(\s+)?(Company(\sname)?\:\s+)?(.+limited)(\s+)?$',
                r'^(\s+)?(.+\:)?(\s+)?(.+ltd)\.?(\s+)?$',
                r'^(\s+)?(.+\:)?(\s+)?(.+ltd.)(\s+)?',
                r'^(\s+)?(.+\:)?(\s+)?(.+,?\sInc\.)\.?(\s+)?$',
                r'^(\s+)?(.+\:)?(\s+)?(.+, llc\.)(\s+)?',
                r'Company\:(\s+)?(.+)(\s+)?',
                r'Manufacturer\:(\s+)?(.+)(\s+)?',
                r'^Company\sand\saddress(\s+)?(.+)(\s+)?',
                r'Name\sof\sCompany(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'^(\s+)?Manufactur(er|ed)?(\s+)?(by)?\:(\s+)(.+)(\s+)?',
                r'Supplier’s details(\s+)(.+)(\s+)?',
                r'^Manufacturer/(distributor|Supplier)(\s+)?(\:)?(\s+)(.+)(\s+)?',
                r'^Manufacturer’s Name(\:)?(\s+)(.+)(\s+)?',
                r'Manufacturer\sor\sDistributor(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'Empresa/Company(\s+)?(\:)?(\s+)?(.+)(\s+)?',
                r'^Supplier( |\s)company( |\s)(\s+)?identification(\:)?(\s+)?(.+)(\s+)?',
                r'Business\sName\:(\s+)?(.+)(\s+)?',
                r'^(\s+)?(Registered\s)?Company(?!/(\s+)?UNDERTAKING)((\s)?(name|identification|/\(Sales office\)))?(\s+)?(\:)?(\s+)?(.+)(\s+)?$',
                r'Supplier\s\(manufacturer/importer/downstream user/distributor\)(\:)?(\s+)?(.+)?(\s+)?',
                r'Supplier(\s+)?\(manufacturer(\s+)?/(\s+)?importer(\s+)?/(\s+)?only(\s+)?representative(\s+)?/(\s+)?downstream(\s+)?user(\s+)?/(\s+)?distributor\)(\:)?(\s+)?(.+)(\s+)?',
                r'Supplier \(importer/only\srepresentative/downstream\suser/distributor\)(\s+)?(.+)(\s+)?',
                r'Headquarters(\:)?(\s+)?(.+)(\s+)?',
                r'^(\s+)?(.)?(\s+)?Supplier(\(Manufacturer\))(\'s\sdetails)?(\s+)?(\:)?(\s+)?(.+)($)',
                r'^(Manufacturer\s)?Address(/Phone No\.)?(\:)?(\s+)?(.+)($)',
                r'^(\s+)?Supplier(\s+)?(\'s\sdetails|(\sof\sthe\ssafety\sdata\ssheet))?(\:)?(\s+)?(.+)(\s+)?$',
                r'^(.+)?Manufacturer((\s)?/(\s)?Supplier|(\s)?/(\s+)?distributor|)?(\s+)?(\:)?(\s+)?(.+)($)',
                r'^(\s+)?NAME\sOF\sMANUFACTURER/SUPPLIER(\s+)?(\:)?(\s+)?(.+)\s+?($)',
                r'Details\sof\sthe\ssupplier.+(\s+)?(\.)?(Name)?(\s+)?(.+)(\s+)',
                r'Supplier\:(\s+)?(.+)(\s+)?',
                r'Distributor(\:)?(\s+)?(.+)(\s+)?',
                r'Company/Undertaking Identification(\s+)?(.+)(\s+)?',
                r'Company name of supplier(\s+)?(\:)(\s+)?(.+)(\s+)?',
                ]
    found = False
    for f in patterns:
        cname = re.search(f, contents, re.IGNORECASE | re.MULTILINE)
        if cname is not None:
            if cname.group(len(cname.groups()) - 1).find("emergency telephone number") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("1.3") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("  ") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("address") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().startswith("supplier") == False \
                    and cname.group(len(cname.groups()) - 1).lower().find("name:") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("see distributor.") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("information:") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("headquarters: ") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("see distributor.") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("manufacturer:") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("’s name ") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("e-mail address") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("representative.") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("phone number") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("/ undertaking") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("suuplier details") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("/manufacturer") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("of supplier") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("identification") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("supplier:") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("telephone") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("or supplier's details") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("company/undertaking identification") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().startswith("data sheet") == False \
                    and cname.group(len(cname.groups()) - 1).lower().find("distributed by") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().startswith("distributor") == False \
                    and cname.group(len(cname.groups()) - 1).lower().startswith("company") == False \
                    and cname.group(len(cname.groups()) - 1).lower().startswith("product name") == False \
                    and cname.group(len(cname.groups()) - 1).lower().find("details of the supplier of the safety") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("e-mail address of person") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().find("/ importer/supplier: ") == -1 \
                    and cname.group(len(cname.groups()) - 1).lower().startswith("1") == False \
                    and cname.group(len(cname.groups()) - 1) != ":" \
                    and cname.group(len(cname.groups()) - 1) != ": " \
                    and cname.group(len(cname.groups()) - 1) != "s " \
                    and cname.group(len(cname.groups()) - 1) != "USA" \
                    and cname.group(len(cname.groups()) - 1).lower().find("street") == -1:
                if cname.group(len(cname.groups()) - 1).find(", Inc.") != -1:
                    result = cname.group(len(cname.groups()) - 1).split(', Inc.')[0] + ", Inc."
                elif cname.group(len(cname.groups()) - 1).find(", LLC.") != -1:
                    result = cname.group(len(cname.groups()) - 1).split(', LLC.')[0] + ", LLC."
                else:
                    result = cname.group(len(cname.groups()) - 1).split(',')[0].replace(': ', '').replace(':', '')
                found = True
                break

    if found is not True:
        emailptrn = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
        email = re.search(emailptrn, contents, re.IGNORECASE)
        if email is not None:
            result = email.group(0).split('@')[1].split('.')[0]
            # print(result)
        else:
            result = "Not found"
    return result


def get_email(contents, pdf_file, type):
    result = ""
    found = False
    emailptrn = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    email = re.search(emailptrn, contents, re.IGNORECASE)
    if email is not None:
        result = email.group(0)
        found = True
    if found is not True:
        result = "Not found"
    return result


def get_comparison_lists(results):
    """
    Helper function to split similarity function results
    into two lists:
        :keylines_with_candidates: keylines from candidate search in their truthy order
        :ordered_keylines: same keylines in order their most similar candidate line
            appear in the document
    """
    keylines_with_candidates = [result['keyline']['line'] for result in results]

    ordered_keylines = [
        result['keyline']['line']
        for result in sorted(results, key=lambda result: result['position'])
    ]
    return keylines_with_candidates, ordered_keylines


def calc_order_score(document_keylines, ordered):
    '''
    Levenstein distance approach on lists:
    Encode each element with single letter
    and calculate the distance
    '''
    mapping = {line: chr(i + 100) for i, line in enumerate(document_keylines)}

    document_keylines_decoded = ''.join([mapping[line] for line in document_keylines])
    ordered_decoded = ''.join([mapping[line] for line in ordered])
    return fuzz.ratio(document_keylines_decoded, ordered_decoded)


def is_ocr_needed(lines, doc_text, language, layout_markers):
    '''Attempt to determine if OCR needed with all the heuristics'''
    # Wrong language usually means broken text layer

    if language not in layout_markers:
        return True

    # Section 3 stands earlier in the document than Section 1
    # This is also a sign of a bad text layer with wrong orderiing
    main_marker = layout_markers[language]['main_marker']
    section_name = layout_markers[language]['section_name']
    broken_layout = False

    if main_marker in doc_text:
        if -1 < doc_text.find(f'{section_name} 3') < doc_text.find(f'{section_name} 1'):
            broken_layout = True

    # Combine conditions together
    ocr_condition = (
        len(lines) < LINES_FOR_OCR or  # too few lines
        broken_layout
    )
    return ocr_condition


def process(pdf_file, keylines, section3_candidates, layout_markers, use_ocr=True):
    '''Main processing function'''
    product_name = ""
    broken_layout = False
    lines, meta = get_document_data(pdf_file)

    # Main OCR condidtion: too few lines or not an english text

    doc_text = ''.join(lines)
    language = get_language(doc_text)

    if is_ocr_needed(meta['all_lines'], doc_text, language, layout_markers):
        if OCR_AVAILABLE and use_ocr:
            # quite a bit of time (may take minutes per file)
            # OCR file inplace and process it again
            print('Not enough text or not english or broken text, starting OCR')
            try:
                ocrmypdf.ocr(pdf_file, pdf_file, force_ocr=True, deskew=False, optimize=0, output_type='pdf')  # noqa
                return process(pdf_file, keylines, section3_candidates, layout_markers, use_ocr=False)
            except Exception as e:
                print(f'OCR failed with error: {str(e)}, results may be poor')
        else:
            raise InvalidTextError(f'File {pdf_file} needs OCR, not enough text or bad text/layout')

    # After OCR is done, we get language specific reference data and start the matching
    # If inferred language doesnt match our set - raise exception
    try:
        specific_keylines = keylines[language]
        specific_section3_candidates = section3_candidates[language]
        specific_section3_anchor = layout_markers[language]['section3_anchor']
    except KeyError:
        raise InvalidTextError(f'File {pdf_file} has invalid language: {language}')

    candidates = get_candidate_lines(specific_keylines, lines, meta)
    mean_distance = np.mean([
        cand['similarity'] for cand in candidates
        if not cand['keyline']['helper_line']
    ])
    order_score = calc_order_score(*get_comparison_lists(candidates))
    final_score = (order_score + mean_distance) / 2
    sds_count = get_possible_sds_count(
        final_score, meta, specific_section3_candidates, specific_section3_anchor
    )
    contents = get_pdf_strings(pdf_file, 1, False, False)
    product_name = get_productname(contents, pdf_file, 1)

    companynamecontents = get_pdf_strings(pdf_file, 2, False, False)
    company_name = get_companyname(companynamecontents, pdf_file, 2)
    if company_name == "Not found":
        companynamecontents = get_pdf_strings(pdf_file, 2, True, False)
        company_name = get_companyname(companynamecontents, pdf_file, 2)

    emailcontents = get_pdf_strings(pdf_file, 3, False, False)
    email = get_email(emailcontents, pdf_file, 3)
    if email == "Not found":
        emailcontents = get_pdf_strings(pdf_file, 3, True, False)
        email = get_email(emailcontents, pdf_file, 3)

    special_char_file = open("special_characters_not_to_trim.txt", "r+")
    special_char_list = special_char_file.readlines()
    special = ""
    for f in special_char_list:
        line = re.sub(r"\n","",f)
        special += line

    product_name_trimmed = re.sub(r"[^\w " + special + "]+", '', product_name, flags=re.UNICODE)
    product_name_trimmed = re.sub(' +', ' ', product_name_trimmed)

    company_name_trimmed = re.sub(r"[^\w " + special + "]+", '', company_name, flags=re.UNICODE)
    company_name_trimmed = re.sub(' +', ' ', company_name_trimmed)

    product_file = open("product_exclusion_list.txt", "r+")
    product_exclusion_list = product_file.readlines()
    product_exclustion_similarity = 0
    for f in product_exclusion_list:
        line = re.sub(r"\n","",f)
        ratio = round(fuzz.ratio(line, product_name_trimmed) / 100, 1)
        if ratio > product_exclustion_similarity:
            product_exclustion_similarity = ratio

    company_file = open("companyname_exclusion_list.txt", "r+")
    company_exclusion_list = company_file.readlines()
    company_exclustion_similarity = 0
    email_exclustion_similarity = 0
    for f in company_exclusion_list:
        line = re.sub(r"\n", "", f)
        ratio = round(fuzz.ratio(line, company_name_trimmed) / 100, 1)
        if ratio > company_exclustion_similarity:
            company_exclustion_similarity = ratio
        ratio_email = round(fuzz.ratio(line, email) / 100, 1)
        if ratio_email > email_exclustion_similarity:
            email_exclustion_similarity = ratio

    df = pd.read_csv('master_result_extracted_data.csv')
    product_name_similarity = 0
    company_name_similarity = 0
    email_similarity = 0

    for i, j in df.iterrows():
        for k in j:
            if k == ntpath.basename(pdf_file):
                prod = str(j[1])
                comp = str(j[2])
                mail = str(j[3])
                product_name_similarity = round(fuzz.ratio(prod, product_name_trimmed) / 100, 1)
                company_name_similarity = round(fuzz.ratio(comp, company_name_trimmed) / 100, 1)
                email_similarity = round(fuzz.ratio(mail, email) / 100, 1)

    return {
        'file': ntpath.basename(pdf_file),
        'language':language,
        'Product_name': product_name,
        'Product_name_trimed':product_name_trimmed,
        'similarity_productnamemost_similar_string_in_product_exclusion_list':product_exclustion_similarity,
        'similarity_productname_master_result_extracted_data': product_name_similarity,
        'Company_name' : company_name,
        'Company_name_trimmed': company_name_trimmed,
        'similarity_companyname_most_similar_string_in_company_exclusion_list': company_exclustion_similarity,
        'similarity_companyname_master_result_extracted_data': company_name_similarity,
        'email' : email,
        'similarity_email_most_similar_string_in_companyname_exclusion_list': email_exclustion_similarity,
        'similarity_email_master_result_extracted_data':email_similarity,
        'order_score': order_score,
        'mean_distance': mean_distance,
        'final_score': final_score,
        'sds_count': sds_count,
    }


def get_possible_sds_count(final_score, meta, last_section_candidates, section3_anchor):
    '''
    Main desicion function.

    Possible SDS/NON SDS calculation and attempt to count concatenated SDSs
    inside big files.

    For relatively huge docs with proper final score lets count probable SDS count.
    We assume these docs as concatenated multi SDS files.
    Approach is simple: count top candidates for last (most representative) keyline
    with really high similarity. As this line can vary,
    we match against several candidates.
    '''
    if final_score < 45:
        # Basic NON SDS Case
        sds_count = 0
    elif final_score >= 45 and meta['page_count'] < MULTI_SDS_MIN_PAGE_COUNT:
        sds_count = 1
    elif final_score >= 45 and meta['page_count'] >= MULTI_SDS_MIN_PAGE_COUNT:
        sds_count = 0  # Because we count all SDSs here
        for position, line in enumerate(meta['all_lines']):
            if section3_anchor in line:
                for candidate_line, min_similarity in last_section_candidates.items():
                    similarity = fuzz.ratio(candidate_line, line)
                    if similarity > min_similarity:
                        # Special cases for bad (but very similar) lines:
                        # Doesnt have quotes in them:
                        # INVALID LINE EXAMPLE: 5.1.3 sds section 3 "composition/information on ingredients"
                        # Doesnt start with specific symbols, like "(" or
                        # "1" (because "11" can be a bad OCR of double quote)
                        # INVALID LINE EXAMPLE: (composition/information on ingredients) .
                        bad_line = (line[0] in ['(', '1']) or ('"' in line)

                        if not bad_line:
                            sds_count += 1
                            continue  # Dont test a line anymore if already matched

        # Edge cases: ratio between page count and sds count cant be very low
        # If we observe 1 or 2 pages per SDS - its definately a layout problem.
        # Good example: B74F61F216D24EB5ABBABA08101EABF6.ashx.pdf, which has
        # all secions repeated as agenda at each page
        if sds_count:
            if meta['page_count'] / sds_count <= 2:
                sds_count = 1

    return sds_count


def print_result(result):
    '''Prints result to console'''
    file = result['file']
    final_score = result['final_score']
    sds_count = result['sds_count']
    output = f'{file} {final_score:.2f}% SDS, {sds_count} possible SDSs detected'
    print(output)


if __name__ == '__main__':

    try:
        pdf = sys.argv[1]  # read file from command line
    except IndexError:
        print('Please enter PDF filename')
        exit()

    keylines, section3_candidates, layout_markers = get_reference_data(KEYLINES_FILENAME)

    try:
        result = process(pdf, keylines, section3_candidates, layout_markers)
    except Exception as e:
        print(f'Exception: {str(e)}')
        exit()

    print_result(result)

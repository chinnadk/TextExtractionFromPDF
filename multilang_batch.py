
import os
import sys
import time

import pandas as pd

from multilang_similarity import (
    KEYLINES_FILENAME,
    InvalidTextError,
    get_reference_data,
    process
)

OUTPUT_FILE = 'output.xlsx'


def process_batch(folder, use_ocr):
    '''Process entire folder and save results as CSV'''
    start = time.time()
    errors = 0

    try:
        pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
        print(f'{len(pdf_files)} PDF files found')
    except FileNotFoundError:
        print('Incorrect folder name')
        exit()

    results = []

    keylines, section3_candidates, layout_markers = get_reference_data(KEYLINES_FILENAME)

    for pdf_file in pdf_files:
        filepath = os.path.join(folder, pdf_file)
        try:
            file_result = process(filepath, keylines, section3_candidates, layout_markers, use_ocr=use_ocr)
            results.append(file_result)
        except InvalidTextError as e:
            print(str(e))
        except RuntimeError as e:
            # Corruted files
            print(f'Exception: {str(e)}, file: {pdf_file}')
            results.append({'file': pdf_file, 'order_score': 'Corrupted PDF'})
            errors += 1

    minutes, seconds = divmod(time.time() - start, 60)
    return results, errors, minutes, seconds


if __name__ == '__main__':
    try:
        folder = sys.argv[1]  # read file from command line
    except IndexError:
        print('Folder is not set, fallback to `./pdfs`')
        folder = './pdfs'

    results, errors, minutes, seconds = process_batch(folder, use_ocr=False)
    print(f'Processed in {minutes} minutes, {seconds:.2f} seconds')
    print(f'Corrupted files that were not processed: {errors}')

    pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)

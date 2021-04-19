### Calculating similarity of PDF content with reference list

Requirements:

1. Linux environment (preferable)
2. Python 3.7+
3. Third-party python packages (see `requirements.txt`)
4. Additional software for OCR (optional, install if OCR is needed):

   - `sudo apt install ocrmypdf` (installation notes for other systems: https://ocrmypdf.readthedocs.io/en/latest/installation.html)

   - `pip install ocrmypdf==11.7.3` - python bindings for OCR

Usage: `python similarity.py [filename]`. File with keywords is module level constant, it can be changed directly inside `similarity.py` file

Batch usage on folder with multiple files: `python batch.py [folder]`, the Excel output file is generated: `output.xlsx`

import numpy as np
import pandas as pd

import os
import shutil
import csv

import html2text
from bs4 import BeautifulSoup

import langid

from typing import List, Tuple

def is_valid(text: str) -> bool:
    """Determines if text is valid.

    The validity of the text is determined by whether the language detected
    is English, and the confidence score (the lower the better). See the doc
    for langid at the following link for more details:
    https://github.com/saffsd/langid.py#probability-normalization  

    Args:
        text: text to validate.
    
    Returns:
        True, if text is valid. Otherwise False.
    """ 
    lang, score = langid.classify(text)
    return (lang == 'en') and (score < -10)

def convert_to_text(html: str) -> str:
    """ Extracts text from HTML."""
    try:
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_anchors = True
        h.ignore_images = True
        h.ignore_emphasis = True
        h.skip_internal_links = True
        h.bypass_tables = True
        text = h.handle(html)
        text = BeautifulSoup(text, 'lxml').get_text()
        return text
    except:
        return ""

def save_as_text(original_path: str,
                 out_path: str,
                 errors: List[Tuple[str, int]],
                 display_friendly: bool = False) -> bool:
    """ Saves extracted text from original HTML.

    Args:
        original_path: filepath for original HTML.
        out_path: filepath for output HTML.
        display_friendly: makes output file display-friendly. Defaults to False.
    
    Returns:
        True if saved successfully, False otherwise.
    """    
    # with open(original_path, 'r', encoding='latin1') as f: #encoding='utf-8', errors='ignore'
    with open(original_path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()
    text = convert_to_text(html)
    if not len(text):
        errors.append((original_path, 2))
        return False
    if not is_valid(text):
        errors.append((original_path, 3))
        return False
    if display_friendly:
        text = text.replace('\n', '<br>')
    with open(out_path, 'w+', encoding='utf-8') as f:
        f.write(text)
    return True

def extract_text_data(in_filepaths: List[str],
                      out_data_dir: str,
                      error_dir: str,
                      display_friendly: bool = False) -> List[int]:
    """ Creates directory of parsed HTML files from HTML files in in_filepaths.

    Parses HTML files in in_filepaths to extract text HTML information, to
    save the simplified HTML page in output directory (out_data_dir). Files that
    have not been able to be parsed or do not adhere to the standards specified
    in this project (coherent basic HTML and in English) are ignored, and these
    file paths and their corresponding error codes are stored in the error
    directory in 'preprocessing.csv'. The error codes are as follows:
        1: incorrect file format (not HTML)
        2: parsing error originating from html2text or BeautifulSoup
        3: basic HTML not considered as valid (not in English or not coherent)

    Args:
        in_filepaths: list of file paths to original HTML files to parse.
        out_data_dir: path to output directory for basic HTML files.
        error_dir: path to error directory.
        display_friendly: makes output files display-friendly. Defaults to False.
    
    Returns:
        List of indices of elements in in_filepaths that have successfully been
        parsed and saved as text.
    """

    success_indices = []
    errors = []
    if not os.path.isdir(out_data_dir):
        os.mkdir(out_data_dir)
    for idx, fp in enumerate(in_filepaths):
        fname = fp.split('/')[-1]
        if fname[-5:] == '.html':
            success = save_as_text(fp, out_data_dir + fname, errors, False)
            if success:
                success_indices.append(idx)
        else:
            errors.append((fp, 1))
    if len(errors):
        error_filename = 'preprocessing.csv'
        with open(error_dir + error_filename, 'w+', encoding='utf-8') as out:
            writer = csv.writer(out, delimiter=',')
            writer.writerows(errors)
    
    print('HTML text extraction complete.')

    return success_indices
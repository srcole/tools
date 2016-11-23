"""
This library contains functions to work with html and text data

1. get_html - get the html from a URL
2. extract_subtext_single - extract text from a long string between two known strings
    2a. _find_text_start_index - find the starting (or ending) indices of a pattern in text
3. extract_subtext_many - extract text between 2 repeated keywords
4. text_split_by - split a string into a list of strings by some pattern
5. text_delete_items - delete patterns from text
"""

import numpy as np
from bs4 import BeautifulSoup
import urllib
import re

def get_html(url):
    r = urllib.urlopen(url).read()
    soup = BeautifulSoup(r,"lxml")
    return soup.prettify()

def extract_subtext_single(text, start_str, end_str = None, extract_len = 100, verbose = True, return_empty = True):
    """Function to extract html text I am interest in (give a start string and an end string, or length),
    tell if 1 or many matches"""
    
    # Find locations of start strings
    match_idx = _find_text_start_index(text, start_str)
    
    # Throw error if more than 1 match found
    if len(match_idx) > 1:
        raise ValueError('More than 1 match found')

    # Clip input text to start_str
    try:
        text = text[int(match_idx):]
    except:
        if return_empty:
            return ''
        else:
            raise ValueError('Probably did not find a string matching the starting str')
        
    if end_str is not None:
        # Find location of end strings
        end_idx = _find_text_start_index(text, end_str)

        # Throw error if more than 1 match found
        if len(end_idx) > 1:
            if verbose:
                print 'More than 1 match found for end_str. Taking first match'
            end_idx = end_idx[0]
        if not end_idx:
            raise ValueError('end_str not found at all')

        # Clip text to end str
        text = text[:int(end_idx)]
    else:
        text = text[:extract_len]
        
    return text
    
def _find_text_start_index(text, str_match, index_save = 'start'):
    re_iter = re.finditer(str_match,text)
    match_idx = np.zeros(0,dtype=int)
    for i, match in enumerate(re_iter):
        if index_save == 'start':
            match_idx = np.append(match_idx,match.start())
        elif index_save == 'end':
            match_idx = np.append(match_idx,match.end())
        else:
            raise ValueError('Invalid entry for index_save')
    return match_idx


def extract_subtext_many(text, start_str, end_str = None, extract_len = 100):
    """Function to extract html text I am interest in (give a start string and an end string, or length)"""
    
    # Find locations of start and end strings
    start_idx = _find_text_start_index(text, start_str)
    
    if end_str is not None:
        end_idx = _find_text_start_index(text, end_str, index_save='end')

        # Make sure they are the same length
        if len(start_idx) != len(end_idx):
            print 'WARNING: lengths of matches do not match', len(start_idx), len(end_idx)
    
    # For each start str, clip it to the next occurrence of the end string
    strs = ['']*len(start_idx)
    for i, idx1 in enumerate(start_idx):
        if end_str is not None:
            # Find closest end index following the start index
            idx2 = end_idx[end_idx- idx1>0][0]
            if not idx2:
                print 'str', i, 'does not have a ending element after it'
            strs[i] = text[idx1:idx2]
        else:
            strs[i] = text[idx1:idx1+extract_len]
        
    return strs


def text_split_by(text, pattern='\n'):
    """Split a long string into lines using a pattern character.
    Then assure each line has content"""
    str_list = np.array([x.strip() for x in text.split(pattern)])
    str_list_nonempty = np.where([not not x for x in str_list])[0]
    return str_list[str_list_nonempty].astype(str).tolist()
    
    
def text_delete_items(text,
                      patterns_delete = ['<.*?>','\n', '  ']):
    for p in patterns_delete:
        cleanr = re.compile(p)
        text = re.sub(cleanr, '', text)
    return text
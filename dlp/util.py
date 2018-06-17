# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:32:40 2018

@author: lucas
"""

# DEFINITION OF THE PATHS

POSTS = "dlp/data/Posts.xml"
LINKS = "dlp/data/PostLinks.xml"
TOKENIZED_POSTS = "dlp/data/posts_df.csv"
TRAIN_SET = "dlp/data/train_df.csv"
TEST_SET = "dlp/data/tests_df.csv"
VAL_SET = "dlp/data/validation_df.csv"
EMBEDDING = "dlp/data/embedding.csv"
DICTIONARY = "dlp/data/dictionary.json"
DICTIONARY_FULL = "dlp/data/dictionary_full.json"

# DEFINITION OF RE

TIME= r'([0-1]?\d|2[0-3])(?::([0-5]?\d))+'
DATE= r'\d\d(\d\d)?(-|/)\d\d(-|/)\d\d(\d\d)?'
VERSION= r'(\d+\.\d+(.\d+)*)|(\S*\d+\S*)+'
PATH= r'(~|\w*|.)?(/\S*)+'
HEXADECIMAL= r'0x[0-9a-fA-F]+'
VARIABLE= r'([a-zA-Z0-9]*_+[a-zA-Z0-9]*)+'
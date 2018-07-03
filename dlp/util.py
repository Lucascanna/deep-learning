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
DICTIONARY_FULL = "dlp/data/dictionary_full.json"

EMBEDDING_UBUNTU = "dlp/data/embedding.csv"
DICTIONARY_UBUNTU = "dlp/data/dictionary.json"

EMBEDDING_WIKI_full_200 = "dlp/data/embedding_wiki_full_200.csv"
DICTIONARY_WIKI_full = "dlp/data/dictionary_wiki_full.json"

DICTIONARY_WIKI_10000 = "dlp/data/dictionary_wiki_10000.json"
EMBEDDING_WIKI_10000_200 = "dlp/data/embedding_wiki_10000_200.csv"

# DEFINITION OF RE

TIME= r'([0-1]?\d|2[0-3])(?::([0-5]?\d))+'
DATE= r'\d\d(\d\d)?(-|/)\d\d(-|/)\d\d(\d\d)?'
VERSION= r'(\d+\.\d+(.\d+)*)|(\S*\d+\S*)+'
PATH= r'(~|\w*|.|https?)?(/\S*)+'
HEXADECIMAL= r'0x[0-9a-fA-F]+'
VARIABLE= r'([a-zA-Z0-9]*_+[a-zA-Z0-9]*)+'
CODE= r'<code>.*</code>'
TAB = r'(\n)+'
NOCHAR = r'(£|\$|%|&|\^|#|§|@|\<|\>|\\)+'
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time

import dlp.util as util
from dlp.data_loader import XMLDataLoader
from dlp.data_filter import XMLDataFilter
from dlp.data_parser import XMLtoPandasParser
from dlp.non_dup_generator import NonDupGenerator
from dlp.train_test_val_generation import TrainTestValGenerator
from dlp.tokenizer import Tokenizer
from bs4 import BeautifulSoup
from nltk import word_tokenize
import re
import spacy
from spacy.tokenizer import Tokenizer 
from selectolax.parser import HTMLParser

xml_loader = XMLDataLoader()
posts_root = xml_loader.load_posts()
links_root = xml_loader.load_links()

#filter the useless rows
xml_filter = XMLDataFilter()
posts_root = xml_filter.filter_posts(posts_root)
links_root = xml_filter.filter_links(links_root)
xml_to_pandas_parser = XMLtoPandasParser()
posts_df = xml_to_pandas_parser.to_dataframe_posts(posts_root)
dup_df = xml_to_pandas_parser.to_dataframe_links(links_root)


soups = [soup for soup in posts_df['Text']]
soup_text = [HTMLParser(soup).root.text() for soup in soups]
nlp= spacy.load("en")
tokenizer = Tokenizer(nlp.vocab)
doc = [[t.lemma_ for t in doc] for doc in tokenizer.pipe(soup_text)]
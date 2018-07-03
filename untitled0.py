#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:56:07 2018

@author: meri
"""

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

xml_loader = XMLDataLoader()
posts_root = xml_loader.load_posts()

# filter the useless rows
xml_filter = XMLDataFilter()
posts_root = xml_filter.filter_posts(posts_root)

# parse the xml into pandas dataframes (dropping useless columns)
xml_to_pandas_parser = XMLtoPandasParser()
posts_df = xml_to_pandas_parser.to_dataframe_posts(posts_root)
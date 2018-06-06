# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:40:09 2018

@author: lucas
"""

import pandas as pd
import numpy as np
import time

from dlp.data_loader import XMLDataLoader
from dlp.data_filter import XMLDataFilter
from dlp.data_parser import XMLtoPandasParser
from dlp.non_dup_generator import NonDupGenerator

def drop_inconsistent_links(posts_df, links_df):
    cond1 = pd.isnull(posts_df.loc[links_df['Post1'].values]['Text'].values)
    cond2 = pd.isnull(posts_df.loc[links_df['Post2'].values]['Text'].values)
    cond = np.logical_not(cond1) & np.logical_not(cond2)
    return links_df[cond]

def main():
    
    start = time.clock()
    
    #laod the data as xml objects
    xml_loader = XMLDataLoader()
    posts_root = xml_loader.load_posts()
    links_root = xml_loader.load_links()

    #filter the useless rows
    xml_filter = XMLDataFilter()
    posts_root = xml_filter.filter_posts(posts_root)
    links_root = xml_filter.filter_links(links_root)
    
    #parse the xml into pandas dataframes (dropping useless columns)
    xml_to_pandas_parser = XMLtoPandasParser()
    posts_df = xml_to_pandas_parser.to_dataframe_posts(posts_root)
    dup_df = xml_to_pandas_parser.to_dataframe_links(links_root)
    
    load_time = time.clock() - start
    print("TIME TO LOAD AND PREPROCESS THE DATA: ", load_time)
    
    #remove inconsistencies in links
    dup_df = drop_inconsistent_links(posts_df, dup_df)
    
    #reduce dimensionality of duplicate dataframe
    dup_indexes = np.random.choice(range(0, dup_df.shape[0]), size=15500, replace=False)
    dup_df = dup_df.iloc[dup_indexes]
    
    #generate non duplicates pairs of posts
    non_dup_generator = NonDupGenerator(dim=dup_df.shape[0], max_index=posts_df.shape[0], posts_df=posts_df)
    non_dup_df = non_dup_generator.generate_non_dup_df()
    
    print(posts_df.describe())
    print(dup_df.describe())
    print(non_dup_df.describe())
    
if __name__ == '__main__':
   main()
    
    
    
    
    
    
    
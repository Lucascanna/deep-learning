# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 12:55:10 2018

@author: lucas
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
from dlp.tokenizer import TokenizerPosts

def drop_inconsistent_links(posts_df, links_df):
    cond1 = pd.isnull(posts_df.loc[links_df['Post1Id'].values]['Tokens'].values)
    cond2 = pd.isnull(posts_df.loc[links_df['Post2Id'].values]['Tokens'].values)
    cond = np.logical_not(cond1) & np.logical_not(cond2)
    return links_df[cond]

def main():
    """
    global posts_df
    global dup_df
    dup_df.columns=['isDuplicate', 'Post1Id', 'Post2Id']
    if type(posts_df.index) == pd.RangeIndex: 
        posts_df.set_index('Id', inplace=True)
    """
    print("Loading data...")
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
    
    
    #tokenization of questions
    start=time.clock()
    print("Tokenizing text..")
    
    inizio = time.clock()
    tokenizer = TokenizerPosts(posts_df)
    posts_df['Tokens'] = tokenizer.tokenize()
    fine = time.clock()
    print('tempo per un post: ', fine-inizio )

    
    tokenization_time= time.clock() - start
    print("TIME TO TOKENIZE THE TEXT: ", tokenization_time)
    print("NEW TOKENS COLUMN IN THE POSTS DATAFRAME: ")
    print(posts_df["Tokens"].head())
    
    posts_df.drop(labels=["Text"], axis=1, inplace=True)
    
    
    #remove ouliers in posts
    mask = posts_df['Tokens'].apply(lambda x : len(x)<=1000 and len(x)>=20).tolist()
    posts_df = posts_df[mask]
    
    #remove inconsistencies in links
    dup_df = drop_inconsistent_links(posts_df, dup_df)
    
    #reduce dimensionality of duplicate dataframe
    dup_indexes = np.random.choice(range(0, dup_df.shape[0]), size=15500, replace=False)
    dup_df = dup_df.iloc[dup_indexes]
    
    #generate non duplicates pairs of posts
    non_dup_generator = NonDupGenerator(dim=dup_df.shape[0], max_index=posts_df.shape[0], posts_df=posts_df, links_df=dup_df)
    non_dup_df = non_dup_generator.generate_non_dup_df()
    print("POSTS DATAFRAME: ", posts_df.shape[0], " samples")
    print(posts_df.head())
    print("DUPLICATED PAIRS DATAFRAME: ", dup_df.shape[0], " samples")
    print(dup_df.head())
    print("NON-DUPLICATED PAIRS DATAFRAME: ", non_dup_df.shape[0], " samples")
    print(non_dup_df.head())
    
    #generate train, test and validation sets
    tvv_generator = TrainTestValGenerator(dup_df, non_dup_df, train_dim=24000, test_dim=6000, val_dim=1000)
    train_df = tvv_generator.generate_train()
    test_df = tvv_generator.generate_test()
    val_df = tvv_generator.generate_val()
    
    print("TRAIN SET: ", train_df.shape[0], " samples")
    print(train_df.head())
    print("Num of positive samples: ", train_df[train_df["isDuplicate"]==1].shape[0])
    print("TEST SET: ", test_df.shape[0], " samples")
    print(test_df.head())
    print("Num of positive samples: ", test_df[test_df["isDuplicate"]==1].shape[0])
    print("VALIDATION SET: ", val_df.shape[0], " samples")
    print(val_df.head())
    print("Num of positive samples: ", val_df[val_df["isDuplicate"]==1].shape[0])
    
    #saving on files
    start= time.clock()
    print("Writing on files...")
    posts_df.to_csv(util.TOKENIZED_POSTS)
    test_df.to_csv(util.TEST_SET)
    train_df.to_csv(util.TRAIN_SET)
    val_df.to_csv(util.VAL_SET)
    
    write_time= time.clock() - start
    print("TIME TO WRITE THE TEXT: ", write_time)
    
main()
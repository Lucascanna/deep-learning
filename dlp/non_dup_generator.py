# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:06:21 2018

@author: lucas
"""

import pandas as pd
import numpy as np

class NonDupGenerator(object):
    
    def __init__(self, dim, max_index, posts_df):
        super(NonDupGenerator, self).__init__()
        self.dim = dim
        self.max_index =max_index
        self.posts_df = posts_df
    
    def map_postId(self, args):
        x1 = self.posts_df.loc[args[0], 'Id']
        x2 = self.posts_df.loc[args[1], 'Id']
        return pd.Series([x1, x2])
    
    
    def generate_random_indexes(self):
        """
        Generate a dataframe of 2 columns and dim rows contaiing random indexes of posts
        """
        random_indexes = pd.DataFrame(np.random.randint(0, self.max_index, size=(self.dim,2)), columns=['Post1Id', 'Post2Id'])
        self.df = random_indexes
        self.df = self.df.apply(self.map_postId, axis=1)
        self.df.columns = ['Post1Id', 'Post2Id']
    
    
    def check_no_pairs_same_index(self):
        self.df = self.df.apply(lambda x : x if x['Post1Id'] != x['Post2Id'] else print("Error: same pair founded", x), axis=1)
    
    
    def check_no_pairs_dup(self, links_df):
        if (pd.merge(links_df, self.df).shape[0]!=0):
            print("Error: duplicate found")
        if (pd.merge(links_df, self.df.rename(index=str, columns={'Post1Id':'Post2Id', 'Post2Id':'Post1Id'})).shape[0]!=0):
            print("Error: duplicate found")
    
    def generate_non_dup_df(self):
        self.generate_random_indexes(self)
        self.check_no_pairs_same_index(self)
        self.check_no_pairs_dup(self)
        return pd.concat([pd.DataFrame(0, index=range(self.dim), columns=['isDuplicate']), self.df], axis=1)
    
    
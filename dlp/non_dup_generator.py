# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:06:21 2018

@author: lucas
"""

import pandas as pd
import numpy as np

class NonDupGenerator(object):
    
    def __init__(self, dim, max_index, posts_df, links_df):
        super(NonDupGenerator, self).__init__()
        self.dim = dim
        self.max_index =max_index
        self.posts_df = posts_df
        self.links_df = links_df
    
    def __map_postId(self, args):
        x1 = self.posts_df.loc[args[0], 'Id']
        x2 = self.posts_df.loc[args[1], 'Id']
        return pd.Series([x1, x2])
    
    
    def __generate_random_indexes(self):
        """
        Generate a dataframe of 2 columns and dim rows contaiing random indexes of posts
        """
        random_indexes = pd.DataFrame(np.random.randint(0, self.max_index, size=(self.dim,2)), columns=['Post1Id', 'Post2Id'])
        self.df = random_indexes       
        self.df = self.df.apply(self.__map_postId, axis=1)
        self.df.columns = ['Post1Id', 'Post2Id']
    
    
    def __check_no_pairs_same_index(self):
        self.df = self.df.apply(lambda x : x if x['Post1Id'] != x['Post2Id'] else print("Error: same pair founded", x), axis=1)
    
    
    def __check_no_pairs_dup(self):
        if (pd.merge(self.links_df, self.df).shape[0]!=0):
            print("Error: duplicate found")
        if (pd.merge(self.links_df, self.df.rename(index=str, columns={'Post1Id':'Post2Id', 'Post2Id':'Post1Id'})).shape[0]!=0):
            print("Error: duplicate found")
    
    def generate_non_dup_df(self):
        self.posts_df.reset_index(inplace=True)
        self.__generate_random_indexes()
        self.posts_df.set_index('Id', inplace=True)
        self.__check_no_pairs_same_index()
        self.__check_no_pairs_dup()
        return pd.concat([pd.DataFrame(0, index=range(self.dim), columns=['isDuplicate']), self.df], axis=1) #add the isDuplicate column
    
    
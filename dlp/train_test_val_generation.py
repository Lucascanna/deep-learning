# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:30:24 2018

@author: lucas
"""

import pandas as pd

class TrainTestValGenerator(object):
    
    def __init__(self, dup_df, non_dup_df, train_dim, test_dim, val_dim):
        super(TrainTestValGenerator, self).__init__()
        assert train_dim + test_dim + val_dim <= dup_df.shape[0] + non_dup_df.shape[0]
        self.df=pd.concat([dup_df, non_dup_df], ignore_index=True)
        self.train_dim=train_dim
        self.test_dim=test_dim
        self.val_dim=val_dim
        self.__shuffle()
        
        
    def __shuffle(self):
        """
        shuffle both datasets of duplicate and non-duplicate posts
        """
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        
    
    def generate_train(self):
        """
        generate training set
        """
        train_start=0
        train_end=self.train_dim
        return self.df[train_start:train_end]
    
    
    def generate_test(self):
        """
        generate test set
        """
        test_start=self.train_dim
        test_end=self.train_dim+self.test_dim
        return self.df[test_start:test_end]
    
    
    def generate_val(self):
        """
        generate validation set
        """
        val_start=self.train_dim+self.test_dim
        val_end=self.train_dim+self.test_dim+self.val_dim
        return self.df[val_start:val_end]
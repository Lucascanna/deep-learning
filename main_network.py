# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 18:16:41 2018

@author: lucas
"""
import numpy as np
import pandas as pd
import time

from dlp.keras_network import ModelBuilder
import dlp.util as util

def main():
    
    print("Reading data from file...")
    start= time.clock()
    
    #read embeddings
    ubuntu_embeddings = np.loadtxt(util.EMBEDDINGS)
    
    #read train, test and validation set
    posts_df= pd.read_csv(util.TOKENIZED_POSTS, index_col=0, converters={"Tokens": lambda x: x.strip("[]").split(", ")})
    
    train_df = pd.read_csv(util.TRAIN_SET, index_col=0)
    test_df = pd.read_csv(util.TEST_SET, index_col=0)
    val_df = pd.read_csv(util.VAL_SET, index_col=0)
    
    train_df = pd.concat([train_df, val_df])
    train_df = train_df[:1500]
    x_1_train = train_df["Post1Id"].values
    x_2_train = train_df["Post2Id"].values
    y_train = train_df["isDuplicate"].values
    
    
    read_time=time.clock()-start
    print("TIME TO READ THE DATA: ", read_time)
    
    print("Computing q_length...")
    #hyperparameters
    q_length = posts_df['Tokens'].loc[train_df['Post1Id'].tolist() + train_df['Post2Id'].tolist()].apply(lambda x : len(x)).max()
    clu = 200
    window_size = 4
    
    print("Training and validating the model...")
    start=time.clock()
    
    model_builder = ModelBuilder(ubuntu_embeddings, q_length, clu, window_size)
    model = model_builder.buildModel()
    model_builder.compileModel()
    train_history = model_builder.trainModel(model, x_1_train, x_2_train, y_train, batch_size=128, num_epochs=50)
    
    train_time= time.clock()-start
    print("TIME TO TRAIN THE MODEL: ", train_time)
    print("HISTORY: ", train_history.history)
    
main()
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:40:09 2018

@author: lucas
"""

import pandas as pd
import time

import dlp.util as util
from dlp.word_embedding import WordEmbedding


def main():
    
    print("Reading posts from file...")
    start= time.clock()
    
    posts_df= pd.read_csv(util.TOKENIZED_POSTS, index_col=0, converters={"Tokens": lambda x: x.strip("[]").split(", ")})
    
    read_time=time.clock()-start
    print("TIME TO READ THE DATA: ", read_time)
    
    train_df= posts_df #[:80000]
    test_df= posts_df[22500:]   #[80000:100000]
    
    del posts_df
    
    #hyperparameters
    vocabulary_size=10000
    skip_window = 2
    embedding_size = 100
    
    print("Data preprocessing...")
    start= time.clock()
    
    #build data (conveert words into indexes)
    word_embedding = WordEmbedding(vocabulary_size, embedding_size, skip_window)
    [data_train, _, _, reversed_dictionary] = word_embedding.emb_build_dataset(train_df['Tokens'], vocabulary_size)
    [data_test, _, _, _] = word_embedding.emb_build_dataset(test_df['Tokens'], vocabulary_size)
    
    #generate samples for the word embeddings
    [target_train, context_train, labels_train] = word_embedding.generate_samples(data_train)
    [target_test, context_test, labels_test] = word_embedding.generate_samples(data_test)
    
    preprocess_time= time.clock()-start
    print("TIME TO PREPREOCESS DATA: ", preprocess_time)
    
    print("Training and validating the model...")
    start=time.clock()
    
    #train and validate the model
    [model, validation_model] = word_embedding.buildmodel()
    model = word_embedding.compileModel(model)
    history = word_embedding.trainModel(model, validation_model, reversed_dictionary, 
                                        target_train, context_train, labels_train, batch_size=64, num_epochs=2)
    
    train_time= time.clock()-start
    print("TIME TO TRAIN THE MODEL: ", train_time)
    print("HISTORY: ", history.history)
    
main()
    

    
    
    
    
    
    
    
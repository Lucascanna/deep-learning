# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 18:16:41 2018

@author: lucas
"""
import numpy as np
import pandas as pd
import time
import json
from skopt import gp_minimize
from skopt.space import Integer
from keras.callbacks import TensorBoard, EarlyStopping
from skopt.utils import use_named_args
import keras.backend as K

from dlp.keras_network import ModelBuilder
import dlp.util as util

def pick_index(word, dictionary):
    try:
        return dictionary[word]
    except KeyError:
        return 0

def words_to_indexes(post, dictionary, q_max):
    ls=[pick_index(word, dictionary) for word in post] 
    delta=q_max-len(ls)
    if delta > 0:
        ls = ls + [0]*delta
    if delta < 0:
        ls = ls[:q_max]
    arr=np.asarray(ls)
    return arr

def build_indexes_dataset(df, posts_df, dictionary, q_length):
    print("First post: ", posts_df["Tokens"].loc[1])
    batch = df.apply(lambda x: pd.Series([x['isDuplicate'],
                                            words_to_indexes(posts_df['Tokens'].loc[x['Post1Id']], dictionary, q_length),
                                            words_to_indexes(posts_df['Tokens'].loc[x['Post2Id']], dictionary, q_length)]), axis=1)
    batch.columns = ['isDuplicate', 'Post1Indexes', 'Post2Indexes']
    
    y_train = batch.as_matrix(columns=['isDuplicate']).astype(np.float32)
    print("Ytrain shape: ", y_train.shape)
    print(y_train[:10])
    print("type :", y_train.dtype)
    
    print(posts_df.loc[df["Post1Id"][0]])
    x_1_train = batch["Post1Indexes"]
    x_1_train_ls = x_1_train.values.tolist()
    x_1_train = np.asarray(x_1_train_ls)
    print("X1train shape: ", x_1_train.shape)
    print(x_1_train[:10, :])
    print("type :", x_1_train.dtype)
    
    x_2_train = batch["Post2Indexes"]
    x_2_train_ls = x_2_train.values.tolist()
    x_2_train = np.asarray(x_2_train_ls)
    print("X2train shape: ", x_2_train.shape)
    print(x_2_train[:10,:])
    print("type :", x_2_train.dtype)
    
    return x_1_train, x_2_train, y_train

def main():
    
    print("Reading data from file...")
    start= time.clock()
    
    #read embeddings
#    model = load_model('model_embedding.h5')
#    layer_dict = dict([(layer.name, layer) for layer in model.layers])
#    embeddings = layer_dict['embedding'].get_weights()[0]

    embeddings = np.loadtxt(util.EMBEDDING_UBUNTU)
    
    #read train, test and validation set
    posts_df= pd.read_csv(util.TOKENIZED_POSTS, index_col=0, converters={"Tokens": lambda x: x.strip("[]").replace("'","").split(", ")})   
    train_df = pd.read_csv(util.TRAIN_SET, index_col=0)
    test_df = pd.read_csv(util.TEST_SET, index_col=0)
    val_df = pd.read_csv(util.VAL_SET, index_col=0)
    
    train_df = pd.concat([train_df, val_df]) 
    #train_df = train_df[:100]

    print("TRAINING SET...")
    print(train_df.head(5))
    #read the dictionary
    with open(util.DICTIONARY_UBUNTU, 'r') as fp:
        dictionary = json.load(fp)
    dictionary = {k.strip("'"): v for k, v in dictionary.items()}
    print(list(dictionary.keys())[:10])
    
    read_time=time.clock()-start
    print("TIME TO READ THE DATA: ", read_time)
    
    #hyperparameters
    clu = 600
    window_size = 3
    
    print("Computing q_length...")
    q_length = posts_df['Tokens'].loc[train_df['Post1Id'].tolist() + train_df['Post2Id'].tolist()].apply(lambda x : len(x)).mean()
    q_length = int(round(q_length))
    x_1_train, x_2_train, y_train = build_indexes_dataset(train_df, posts_df, dictionary, q_length)
    
    print("Training and validating the model...")
    start=time.clock()

    model_builder = ModelBuilder(embeddings, q_length, clu, window_size)
    model = model_builder.buildModel()
    model_builder.compileModel(model)
    print(model.summary())
    train_history = model_builder.trainModel(model, x_1_train, x_2_train, y_train, batch_size=128, num_epochs=200)

    train_time= time.clock()-start
    print("TIME TO TRAIN THE MODEL: ", train_time)
    print("HISTORY: ", train_history.history)
    
main()
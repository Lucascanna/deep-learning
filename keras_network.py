# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:31:34 2018

@author: lucas
"""

import numpy as np
from keras.models import Model
from keras.layers import Input,Embedding, Conv1D, Activation, Dot
import keras.backend as K

#%%

#parameters of the network
q_length=10
vocabulary_size=70
embedding_size=5
window_size=3
clu=7

#parameters of the training
batch_size=32
num_epochs=5

#fake data
x_1_train= np.arange(64*q_length, dtype=np.int32).reshape((64, q_length))
x_2_train= np.arange(64*q_length, dtype=np.int32).reshape((64, q_length))
y_train= np.concatenate((np.ones(32), np.ones(32)))
ubuntu_embeddings=np.ones((vocabulary_size,embedding_size), dtype=np.float32)

def ubuntu_initializer(shape, dtype=None):
    global ubuntu_embeddings
    assert shape==ubuntu_embeddings.shape
    return ubuntu_embeddings
    

def buildModel(vocabulary_size, q_length, embedding_size, clu, window_size):
    q_1= Input(shape=(q_length,), dtype='int32')
    q_2= Input(shape=(q_length,), dtype='int32')    
    
    lookup_layer_1=Embedding(vocabulary_size, embedding_size, input_length=q_length, embeddings_initializer=ubuntu_initializer)(q_1)
    lookup_layer_2=Embedding(vocabulary_size, embedding_size, input_length=q_length, embeddings_initializer=ubuntu_initializer)(q_2)
    
    conv1d=Conv1D(filters=clu, kernel_size=window_size, activation='tanh')
    conv_layer_1=conv1d(lookup_layer_1)
    conv_layer_2=conv1d(lookup_layer_2)
    
    sum_layer_1=K.sum(conv_layer_1, axis=1)
    sum_layer_2=K.sum(conv_layer_2, axis=1)
    
    activation_1=Activation('tanh')(sum_layer_1)
    activation_2=Activation('tnah')(sum_layer_2)
    
    similarity_layer= Dot(normalize=True)([activation_1,activation_2])
    
    threshold= K.constant(0.5, dtype='float32', shape=q_1.shape)
    predictions= K.cast(K.greater_equal(similarity_layer, threshold), dtype='int32')
    
    return Model(inputs=[q_1, q_2], outputs=predictions)

def compileModel(model):
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])    
    return model

def trainModel(model, x_1_train, x_2_train, labels, batch_size, num_epochs):
    model.fit(x=[x_1_train, x_2_train], y=labels, batch_size=batch_size, epochs=num_epochs)
    
def main():
    model=buildModel(vocabulary_size, q_length, embedding_size, clu, window_size)
    model=compileModel(model)
    train_history= trainModel(model, x_1_train, x_2_train, y_train, batch_size, num_epochs)
    print(train_history.history)

main()


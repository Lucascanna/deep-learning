# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:31:34 2018

@author: lucas
"""

import numpy as np
from keras.models import Model
from keras.layers import Input,Embedding, Conv1D, Activation, Dot, Lambda
from keras.callbacks import TensorBoard
import keras.backend as K
import tensorflow as tf


#%%

#parameters of the network
q_length=10
vocabulary_size=1000
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

class ModelBuilder(object):
    
    def __init__(self, embeddings, q_length, clu, window_size):
        super(ModelBuilder, self).__init__()
        self.embeddings=embeddings
        self.vocabulary_size=embeddings.shape[0]
        self.embedding_size=embeddings.shape[1]
        self.q_length=q_length
        self.clu=clu
        self.window_size=window_size

    def embeddings_initialize(self, shape, dtype=None):
        assert shape==self.emebddings.shape
        return self.embeddings
        
    
    def buildModel(self):
        q_1= Input(shape=(q_length,), dtype='int32')
        q_2= Input(shape=(q_length,), dtype='int32')    
        
        lookup_layer_1=Embedding(vocabulary_size, embedding_size, input_length=q_length, embeddings_initializer=self.embeddings_initilize)(q_1)
        lookup_layer_2=Embedding(vocabulary_size, embedding_size, input_length=q_length, embeddings_initializer=self.embeddings_initialize)(q_2)
        
        conv1d=Conv1D(filters=clu, kernel_size=window_size, activation='tanh')
        conv_layer_1=conv1d(lookup_layer_1)
        conv_layer_2=conv1d(lookup_layer_2)
        
        sum_layer_1=Lambda(lambda x: K.sum(x,axis=1))(conv_layer_1)
        sum_layer_2=Lambda(lambda x: K.sum(x,axis=1))(conv_layer_2)
        
        activation_1=Activation('tanh')(sum_layer_1)
        activation_2=Activation('tanh')(sum_layer_2)
        
        similarity_layer= Dot(axes=1, normalize=True, name='similarity')([activation_1,activation_2])
        
        predictions = Lambda(lambda x: K.cast(x>=0.5, dtype='int32'), name='predictions')(similarity_layer)
        
        return Model(inputs=[q_1, q_2], outputs=[similarity_layer, predictions])
    
    def compileModel(model):
        model.compile(loss={'similarity' : 'mean_squared_error'},
                      optimizer='adam',
                      metrics={'predictions' : 'accuracy'})   

    def trainModel(model, x_1_train, x_2_train, labels, batch_size, num_epochs):
        sess = tf.Session()
        tf.summary.FileWriter('./logs/', sess.graph)
        # tensorboard --logdir=logs  for executing TensorBoard 
        # localhost:6006 to view the TensorBoard
        tensorboard = TensorBoard(log_dir='./logs/', histogram_freq=0,
                              write_graph=True, write_images=True)
        return model.fit(x=[x_1_train, x_2_train], 
                         y=labels, 
                         batch_size=batch_size, 
                         epochs=num_epochs,
                         validation_split=0.04,
                         callbacks = [tensorboard])




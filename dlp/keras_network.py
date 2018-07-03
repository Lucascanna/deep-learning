# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:31:34 2018

@author: lucas
"""

import time
from keras.models import Model
from keras.layers import Input,Embedding, Conv1D, Conv2D, Activation, Dot, Lambda, Permute, Concatenate
import keras.backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping


#%%
class ModelBuilder(object):

    def __init__(self, embeddings, q_length, x_1_train, x_2_train, y_train):
        super(ModelBuilder, self).__init__()
        self.embeddings=embeddings
        self.vocabulary_size=embeddings.shape[0]
        self.embedding_size=embeddings.shape[1]
        self.q_length=q_length
        self.x1 = x_1_train
        self.x2 = x_2_train
        self.y = y_train
        self.pools = [tf.reduce_max, tf.reduce_min, tf.reduce_mean]
        #self.clu=clu
        #self.window_size=window_size

    def embeddings_initialize(self, shape, dtype=None):
        assert shape == self.embeddings.shape
        return self.embeddings

    def convolution_words(self, input, window_size):
        out_a = []
        for pooling in self.pools:
            conv1d = Conv1D(filters=self.embedding_size, kernel_size=window_size, activation='tanh', padding='valid')
            conv = conv1d(input)
            pool = pooling(conv, axis=1)
            out_a.append(pool)

        return Concatenate()(out_a)

    def convolutional_embeddings(self, input, window_size):
        out_b = []
        for pooling in self.pools:
            conv = Permute((2,1))(input)
            conv = Conv1D(filters=20, kernel_size=window_size, padding='valid', activation='tanh')(conv)
            pool = pooling(conv, axis=1)
            out_b.append(pool)
        return Concatenate()(out_b)

    def buildModel(self, clu, window_size):
        q_1= Input(shape=(self.q_length,), dtype='int32')
        q_2= Input(shape=(self.q_length,), dtype='int32')    
        
        lookup=Embedding(self.vocabulary_size, self.embedding_size, input_length=self.q_length, embeddings_initializer=self.embeddings_initialize, trainable=False)
        lookup_layer_1= lookup(q_1)
        lookup_layer_2= lookup(q_2)

        # q1_emb and q2_emb (?, q_length * embedding)
        #expand = Lambda(lambda x: K.expand_dims(x, axis=-1))
        #expand = Reshape(target_shape=(self.q_length * self.embedding_size, 1))
        #expanded_layer_1 = Reshape(target_shape=(self.q_length * self.embedding_size, 1))(lookup_layer_1)
        #expanded_layer_2 = Reshape(target_shape=(self.q_length * self.embedding_size, 1))(lookup_layer_2)
        #print("Expand shape: ", expanded_layer_1.shape)

        conv1d = Conv1D(filters=clu, kernel_size=window_size, activation='tanh', padding='valid')
        conv_layer_1 = conv1d(lookup_layer_1)
        conv_layer_2 = conv1d(lookup_layer_2)


        # conv2d=Conv2D(filters=self.clu, kernel_size=(self.window_size, self.embedding_size), activation='tanh', padding='valid')
        # conv_layer_1=conv2d(expanded_layer_1)
        # conv_layer_2=conv2d(expanded_layer_2)
        # print("Conv shape: ", conv_layer_1.shape)

        sum_layer = Lambda(lambda x: K.sum(x,axis=1))
        sum_layer_1=sum_layer(conv_layer_1)
        sum_layer_2=sum_layer(conv_layer_2)

#        reshape_layer = Reshape(target_shape=(self.clu,))
#        reshape_layer_1 = reshape_layer(sum_layer_1)
#        reshape_layer_2 = reshape_layer(sum_layer_2)
#        print("Reshape shape: ", reshape_layer_1.shape)


        activation_layer = Activation('tanh')
        activation_1= activation_layer(sum_layer_1)
        activation_2=activation_layer(sum_layer_2)

        similarity_layer = Dot(axes=1, normalize=True, name='similarity')([activation_1,activation_2])
        
        #predictions = Lambda(lambda x: K.cast(x>=0.5, dtype='float32'), name='predictions')(similarity_layer)
        return Model(inputs=[q_1, q_2], outputs=similarity_layer)

    def compileModel(self, model):
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def trainModel(self,model, x_1_train, x_2_train, labels, batch_size, num_epochs):
        sess = tf.Session()
        tf.summary.FileWriter('./logs/', sess.graph)
        # tensorboard --logdir=logs  for executing TensorBoard 
        # localhost:6006 to view the TensorBoard
        modelid = time.strftime("%Y%m%d%H%M%S")
        tensorboard = TensorBoard(log_dir='./logs/'+modelid, histogram_freq=0,
                              write_graph=True, write_images=True)
        early_stopping = EarlyStopping(patience=20)
        return model.fit(x=[x_1_train, x_2_train], 
                         y=labels, 
                         batch_size=batch_size, 
                         epochs=num_epochs,
                         validation_split=0.04,
                         callbacks = [tensorboard, early_stopping])

    def log_dir_name(self, window_size, clu):
        # The dir-name for the TensorBoard log-dir.
        modelid = time.strftime("%Y%m%d%H%M%S")
        s = "wind_{0}_clu_{1}_{2}/"

        # Insert all the hyper-parameters in the dir-name.
        log_dir = s.format(window_size,
                           clu, modelid)

        return log_dir








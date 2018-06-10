# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 09:55:46 2018

@author: lucas
"""

import tensorflow as tf
import numpy as np
#%%
#NETWORK ARCHITECTURE

q_length=10
vocabulary_size=50
embedding_size=5
window_size=3
clu=7


#input layer
x_1 = tf.placeholder(tf.int32, shape=[None, q_length])
x_2 = tf.placeholder(tf.int32, shape=[None, q_length])

#output layer
#y = tf.placeholder(tf.int32, shape=[None,1])

#word embeddings
ubuntu_embeddings=np.ones((vocabulary_size,embedding_size), dtype=np.float32)
W0 = tf.Variable(ubuntu_embeddings)

#lookup layers
lookup_layer_1= tf.nn.embedding_lookup(W0, x_1)
lookup_layer_2= tf.nn.embedding_lookup(W0, x_1)

#convolutional layer
filters= tf.Variable(np.ones((window_size, embedding_size, clu), dtype=np.float32))
conv_layer_1= tf.nn.conv1d(
        lookup_layer_1, 
        filters,
        stride=1,
        padding='VALID',
        data_format='NHWC',
        use_cudnn_on_gpu=False)
conv_layer_2= tf.nn.conv1d(
        lookup_layer_2, 
        filters,
        stride=1,
        padding='VALID',
        data_format='NHWC',
        use_cudnn_on_gpu=False)

#sum layer
sum_layer_1= tf.nn.tanh(tf.reduce_sum(conv_layer_1, axis=1))
sum_layer_2= tf.nn.tanh(tf.reduce_sum(conv_layer_2, axis=1))

#normalization layer
l2_norm_1= tf.sqrt(tf.reduce_sum(tf.square(sum_layer_1), axis=-1, keep_dims=True))
l2_norm_2= tf.sqrt(tf.reduce_sum(tf.square(sum_layer_2), axis=-1, keep_dims=True))
norm_layer_1= sum_layer_1 / l2_norm_1
norm_layer_2= sum_layer_2 / l2_norm_2

#similarity layer
score_layer = tf.diag_part(tf.tensordot(norm_layer_1, tf.transpose(norm_layer_2), axes=1))


#%%
init_op = tf.global_variables_initializer()
x_1_batch= np.arange(50, dtype=np.int32).reshape((5,q_length))
x_2_batch= np.arange(50, dtype=np.int32).reshape((5,q_length))

with tf.Session() as sess:
    #initialize the variables
    sess.run(init_op)
    score=sess.run(score_layer, feed_dict={x_1: x_1_batch, x_2: x_2_batch})
    print(score)


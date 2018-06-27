#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:19:37 2018

@author: meri
"""

from keras.layers import Layer
import tensorflow as tf

class NCE_Loss(Layer):
    def __init__(self, num_sampled, num_classes, mode, **kwargs):
        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.mode = mode
        super(NCE_Loss, self).__init__(**kwargs)

    def build(self, input_shape):
        dense_shape, classes_shape = input_shape
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.num_classes, dense_shape[1]),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                      shape=(self.num_classes,),
                                      initializer='uniform',
                                      trainable=True)  # Maybe zero

        super(NCE_Loss, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs_and_labels):
        inputs, labels = inputs_and_labels
        if self.mode == "train":
            loss = tf.nn.sampled_softmax_loss(
                weights=self.kernel,
                biases=self.bias,
                labels=labels,
                inputs=inputs,
                num_sampled=self.num_sampled,
                num_classes=self.num_classes,
                num_true=1)

        elif self.mode == "eval":
            logits = tf.matmul(inputs, tf.transpose(self.kernel))
            logits = tf.nn.bias_add(logits, self.bias)
            labels_one_hot = tf.one_hot(labels, self.num_classes)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels_one_hot,
                logits=logits)

        return loss

    def compute_output_shape(self, input_shape):
        dense_shape, classes_shape = input_shape
        return (dense_shape[0], )
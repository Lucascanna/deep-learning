# -*- coding: utf-8 -*-
import numpy as np
from keras.callbacks import Callback


class ValidationCallback(Callback):
    
    def __init__(self, vocabulary_size, validation_model, reverse_dictionary, validation_set):
        
        super(ValidationCallback, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.validation_model = validation_model
        self.reverse_dictionary = reverse_dictionary
        self.validation_set = validation_set
        
    def __get_sim(self, valid_word_idx):
        
        sim = np.zeros((self.vocabulary_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(self.vocabulary_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = self.validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
    
    def run_sim(self):
        
        validation_size = self.validation_set.shape[0]
        
        for i in range(validation_size):
            valid_word = self.reverse_dictionary[self.validation_set[i]]
            top_k = 8  # number of nearest neighbors
            sim = self.__get_sim(self.validation_set[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
        
    def on_epoch_begin(self, epoch, logs):
        if (epoch+1) % 5 == 0:
            self.run_sim()

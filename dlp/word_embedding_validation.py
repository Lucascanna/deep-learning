# -*- coding: utf-8 -*-
import numpy as np
from keras.callbacks import Callback


class ValidationCallback(Callback):
    
    def __init__(self, vocabulary_size, validation_model, reverse_dictionary):
        
        super(ValidationCallback, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.validation_model = validation_model
        self.reverse_dictionary = reverse_dictionary
        self.validation_indexes = np.random.choice(200, 16, replace=False)
        
    def __get_sim(self, valid_word_idx):
        
        sim = np.zeros((self.vocabulary_size,)) #column vector of zeros of vocabulary_size
        in_arr1 = np.zeros((1,)) 
        in_arr2 = np.zeros((1,))
        for i in range(self.vocabulary_size):
            in_arr1[0,] = valid_word_idx #arr1 set to validation word index
            in_arr2[0,] = i #arr2 set to i-th word in the vocabulary
            #compute the similarity between validation word index and i-th word of vocabulary
            out = self.validation_model.predict_on_batch([in_arr1, in_arr2]) 
            sim[i] = out #set the similarity of word i-th to out
        return sim #return the similarity vector of validation word against all the word in the vocabulary
    
    def run_sim(self):
        
        validation_size = self.validation_indexes.shape[0]
        
        for i in range(validation_size):
            valid_word = self.reverse_dictionary[self.validation_indexes[i]] # get the validation word of index i
            top_k = 8  # number of nearest neighbors
            sim = self.__get_sim(self.validation_indexes[i]) #get the similarity vector of valid word against all the word in the vocabulary
            nearest = (-sim).argsort()[1:top_k + 1] # take the vector of the k nearest words
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
        
    def on_epoch_begin(self, epoch, logs):
        if (epoch+1) % 5 == 0:
            self.run_sim()

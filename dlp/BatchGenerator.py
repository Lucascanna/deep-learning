import numpy as np
from keras.preprocessing.sequence import pad_sequences
import time

class KerasBatchGenerator(object):

    def __init__(self, x1, x2, y,  batch_size):
        self.x1 = x1
        self.x2 = x2
        self.y=y
        self.batch_size = batch_size
        self.current_idx = 0

    def generate(self):
        while True:
            if self.current_idx + self.batch_size >= len(self.x1):
                # reset the index back to the start of the data set
                self.current_idx = 0

            x_1 = pad_sequences(self.x1[self.current_idx:self.current_idx+self.batch_size], padding='post')
            x_2 = pad_sequences(self.x2[self.current_idx:self.current_idx + self.batch_size], padding='post')

            y = self.y[self.current_idx:self.current_idx+self.batch_size]
            yield ([x_1, x_2], y)

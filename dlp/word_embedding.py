import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import collections
from dlp.word_embedding_validation import ValidationCallback
import json
import dlp.util as util


class WordEmbedding(object):
    
    def __init__(self, vocabulary_size, embedding_size, skip_window):
        
        super(WordEmbedding, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.skip_window = skip_window
        self.embedding_size = embedding_size
        
    def emb_build_dataset(self, posts, vocabulary_size):
    
        flat_posts = [word for post in posts for word in post]
        #create a list of tuples (word, count) sorted by count
        count = [["UNK", -1]]
        count.extend(collections.Counter(flat_posts).most_common(self.vocabulary_size - 1))
        
        #give a unique index to each word using a dictionary
        dictionary = dict()
        for word, _ in count:
            dictionary[word]=len(dictionary)
        #reverse the dictionary
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        
        #create a list, where, for each word in the corpus there is the corresponding index
        data = list()
        unk_count = 0
        for word in flat_posts:
            if word in dictionary:
                index = dictionary[word]
            else:
                index=0
                unk_count +=1
            data.append(index)
        
        count[0][1] = unk_count
        with open(util.DICTIONARY_FULL, 'w+') as fp:
            json.dump(dictionary, fp)
        return data, count, dictionary, reversed_dictionary
    
    def generate_samples(self, data):

        sampling_table = sequence.make_sampling_table(self.vocabulary_size)
        couples, labels = skipgrams(data, self.vocabulary_size, window_size=self.skip_window, sampling_table=sampling_table)
        word_target, word_context = zip(*couples)
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")
        
        return [word_target, word_context, labels]
        
    def buildmodel(self):

        # input layer (note that we don't explicitly need the one-hot style matrix, but only a vector with the indexes of the words)
        input_target = Input(shape=(1,), dtype='float32')
        input_context = Input(shape=(1,), dtype='float32')
        
        # embedding layer
        embedding = Embedding(self.vocabulary_size, self.embedding_size, name='embedding')
        target = embedding(input_target)
        target = Reshape((self.embedding_size, 1))(target) #column vector
        context = embedding(input_context)
        context = Reshape((self.embedding_size, 1))(context) #column vector
        
        # now perform the dot product operation to get a similarity measure
        dot_product = Dot(axes=1)([target, context])
        dot_product = Reshape((1,))(dot_product)
        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid', name='output')(dot_product)
        
        # setup a cosine similarity operation for the validation
        similarity = Dot(axes=0, normalize=True)([target, context])
        # create the primary training model
        model = Model(input=[input_target, input_context], output=[output])
        validation_model = Model(input=[input_target, input_context], output=[similarity])
        
        return model, validation_model
    
    def compileModel(self, model):
        model.compile(loss={'output' : 'binary_crossentropy'},
                      optimizer='rmsprop',
                      metrics= ['accuracy'])    
        return model
    
    def trainModel(self, model, validation_model, reverse_dictionary, target, context, labels, batch_size, num_epochs):
        callback_list = [ValidationCallback(self.vocabulary_size, validation_model, reverse_dictionary),
                         EarlyStopping(monitor='val_loss', patience=5),
                         ModelCheckpoint(
                                 filepath='model_embedding.h5',
                                 monitor='val_loss',
                                 save_best_only=True),
                         TensorBoard(log_dir='./logs_embedding/', histogram_freq=0,
                              write_graph=True, write_images=True, embeddings_layer_names=['embedding'], embeddings_freq=5)]
        return model.fit(x=[target, context], y=labels, batch_size=batch_size, validation_split=0.12, epochs=num_epochs, callbacks=callback_list)
    




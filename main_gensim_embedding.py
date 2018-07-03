import os
from keras.layers import Input, Embedding, Dot
from keras.models import Model
import numpy as np
from gensim.models import Word2Vec
import logging
import pandas as pd
import dlp.util as util
import time
import json

def create_embedding_matrix_dictionary(model, vector_dim, vocabulary_size):

    embedding_matrix = np.zeros((vocabulary_size, vector_dim))
    dictionary = {}
    dictionary['UNK'] = 0
    for i in range(vocabulary_size-1):
        embedding_vector = model.wv[model.wv.index2word[i]]
        word = model.wv.index2word[i]
        dictionary[word] = len(dictionary)
        if embedding_vector is not None:
            embedding_matrix[i+1] = embedding_vector

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return embedding_matrix, dictionary, reversed_dictionary

def save(embeddings, dictionary):
    with open(util.DICTIONARY_GENSIM_10000, 'w+') as fp:
        json.dump(dictionary, fp)
    np.savetxt(util.EMBEDDING_GENSIM_10000_200, embeddings)

def keras_model(embedding_matrix, reversed_dictionary):
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    # input words - in this case we do sample by sample evaluations of the similarity

    valid_word = Input((1,), dtype='int32')
    other_word = Input((1,), dtype='int32')
    # setup the embedding layer
    embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix])
    embedded_a = embeddings(valid_word)
    embedded_b = embeddings(other_word)
    similarity = Dot(axes=2, normalize=True)([embedded_a, embedded_b])
    # create the Keras model
    k_model = Model(input=[valid_word, other_word], output=similarity)

    def get_sim(valid_word_idx, vocab_size):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = k_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

    # now run the model and get the closest words to the valid examples
    for i in range(1, valid_size):
        valid_word = reversed_dictionary[i]
        top_k = 8  # number of nearest neighbors
        sim = get_sim(valid_examples[i], len(reversed_dictionary))
        nearest = (-sim).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = reversed_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)

    return k_model

def gensim_model():

    start = time.clock()
    posts_df = pd.read_csv(util.TOKENIZED_POSTS, index_col=0,
                           converters={"Tokens": lambda x: x.strip("[]").replace("'", "").split(", ")})
    read_time = time.clock() - start
    print("TIME TO READ THE DATA: ", read_time)

    sentences = posts_df['Tokens'].values.tolist()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    embedding_size = 200
    vocabulary_size = 10000
    negative = 10
    model = Word2Vec(sentences, iter=10, min_count=10, size=embedding_size, negative=negative, workers=6)
    #model.save('dlp/data/' + "gensim_model")
    embedding_matrix, dictionary, reversed_dictionary = create_embedding_matrix_dictionary(model, vector_dim=embedding_size, vocabulary_size=vocabulary_size)
    save(embedding_matrix, dictionary)
    k_model = keras_model(embedding_matrix, reversed_dictionary)

def main():

    gensim_model()

main()
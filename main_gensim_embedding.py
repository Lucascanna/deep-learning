import os
from keras.layers import Input, Embedding, Dot
from keras.models import Model
import numpy as np
from gensim.models import Word2Vec
import logging
import pandas as pd
import dlp.util as util
import time

def create_embedding_matrix(model, vector_dim):

    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def keras_model(embedding_matrix, wv):
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
    for i in range(valid_size):
        valid_word = wv.index2word[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        sim = get_sim(valid_examples[i], len(wv.vocab))
        nearest = (-sim).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = wv.index2word[nearest[k]]
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
    print(sentences[0:2])
    # model = Word2Vec(sentences, iter=50, min_count=10, size=300, workers=4)
    # model.save('dlp/data/' + "gensim_model")
    # embedding_matrix = create_embedding_matrix(model, vector_dim=300)
    # k_model = keras_model(embedding_matrix, model.wv)

def main():

    gensim_model()

main()
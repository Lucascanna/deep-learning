# -*- coding: utf-8 -*-

from dlp.word_embedding import WordEmbedding
import numpy as np

global posts_df

vocabulary_size=233
skip_window = 2
embedding_size = 100
validation_size = 10
validation_window = 20
word_embedding = WordEmbedding(vocabulary_size, embedding_size, skip_window)
[data, count, dictionary, reversed_dictionary] = word_embedding.emb_build_dataset(posts_df.iloc[0:5]['Tokens'], vocabulary_size)
[word_target, word_context, labels] = word_embedding.generate_samples(data)
[model, validation_model] = word_embedding.buildmodel()
model = word_embedding.compileModel(model)
validation_set = np.random.choice(validation_window, validation_size, replace=False)
history = word_embedding.trainModel(model, validation_model, reversed_dictionary, 
                                    validation_set, word_target, word_context, labels, batch_size=32, num_epochs=11)
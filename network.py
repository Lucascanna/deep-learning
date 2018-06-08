# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:12:09 2018

@author: lucas
"""

#%% NETWORK ARCHITECTURE

posts_df.set_index('Id', inplace=True)
q_length = posts_df['Tokens'].loc[train_df['Post1'].tolist() + train_df['Post2'].tolist()].apply(lambda x : len(x)).max()
batch_size = 100
epochs = 1

#input layer: [batch_size x 2 x q_length]
x = tf.placeholder(tf.int32, shape=[None, 2, q_length])
y = tf.placeholder(tf.int32, shape=[None,1])

#weigths between input and first hidden layer
W0 = tf.Variable(ubuntu_embeddings)

#output first layer: [batch_size x 2 x q_length x embedding_size]
q_emb = tf.nn.embedding_lookup(W0, x)

#hyperparameters
window_size = 3
clu = 10

#convolutional layer: [batch_size x 2 x q_length x clu]
conv_layer = tf.layers.conv2d(inputs=q_emb, filters=clu, kernel_size=[window_size, embedding_size], activation=tf.tanh, padding='same')

#returns the question-wide vector representation [batch_size x 2 x clu].
sum_layer = tf.nn.tanh(tf.reduce_sum(conv_layer, axis=2))

#SIMILARITY SCORE
#normalization: question-wide vectors are normalized, i.e. each element is divided by the L2-norm of the vector
l2_norm = tf.sqrt(tf.reduce_sum(tf.square(sum_layer), axis=-1, keep_dims=True))
normalization_layer = sum_layer / l2_norm
#computing the similarity score between vector representation of q1 and q2: [batch_size]
r_q1, r_q2 = tf.split(normalization_layer, 2, 1)
r_q1 = tf.squeeze(r_q1, axis=1)
r_q2 = tf.squeeze(r_q2, axis=1)
score_layer = tf.tensordot(r_q1, tf.transpose(r_q2), axes=1)
score_layer_exp = tf.expand_dims(score_layer, 1)

#computing prediction and accuracy
treshold = tf.constant(np.ones(shape=(batch_size,1))/2, dtype=tf.float32)
prediction = tf.cast(tf.greater_equal(score_layer, treshold), tf.int32)

correct_prediction = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

#loss function: mean squared error
loss=tf.losses.mean_squared_error(y, score_layer)

#minimize the loss using gradient descent
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
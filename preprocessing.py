#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import collections
import random
import math
import tensorflow as tf
from bs4 import BeautifulSoup
from nltk import word_tokenize
import re

#%% READING THE XMLs FILES

#from an xml element containing a post to a dictionary
def parse_post(elem):
    
        post = dict()
        postId = elem.attrib.get('Id')
        post['Id'] = postId
        body = elem.attrib.get('Body')
        title = elem.attrib.get('Title')
        post['Text'] = title + body
        return post
    
#from an xml element containing link between two duplicated posts to a dictionary
def parse_link(elem):
    link = dict()
    link['Post1'] = elem.attrib.get('PostId')
    link['Post2'] = elem.attrib.get('RelatedPostId')
    link['Duplicate'] = 1
    return link
    
# parsing Posts.xml into a pandas dataframe
root_posts = ET.parse('Posts.xml').getroot()
# filter out posts that are not questions
posts_ls = [ parse_post(elem) for elem in root_posts.findall("./row[@PostTypeId='1']")]
posts_df = pd.DataFrame.from_records(posts_ls)
posts_df.set_index('Id', inplace=True)

# parsing PostLinks.xml into a pandas dataframe
root_links = ET.parse('PostLinks.xml').getroot()
dup_ls = [ parse_link(elem) for elem in root_links.findall("./row[@LinkTypeId='3']") ]
dup_df = pd.DataFrame.from_records(dup_ls)

#check links with Posts that are questions
cond1 = pd.isnull(posts_df.loc[dup_df['Post1'].values]['Text'].values)
cond2 = pd.isnull(posts_df.loc[dup_df['Post2'].values]['Text'].values)
cond = np.logical_not(cond1) & np.logical_not(cond2)
dup_df = dup_df[cond]

#reduce dimensionality od duplicate dataframe
dup_indeces = np.random.choice(range(0, dup_df.shape[0]), size=15500, replace=False)
dup_df = dup_df.iloc[dup_indeces]

del cond
del cond1
del cond2

#%% GENERATION OF NON-DUPLICATED POSTS

posts_df.reset_index(inplace=True)
# generate a dataframe of random pairs of post indexes
non_dup_df = pd.DataFrame(np.random.randint(0, posts_df.shape[0], size=(dup_df.shape[0],2)), columns=['Post1', 'Post2'])
# substitute indexes with PostIds
def map_postId(args):
    x1 = posts_df.loc[args[0], 'Id']
    x2 = posts_df.loc[args[1], 'Id']
    
    return pd.Series([x1, x2])

non_dup_df = non_dup_df.apply(map_postId, axis=1)
non_dup_df.columns = ['Post1', 'Post2']

# check that there are no pairs with the same postId
non_dup_df = non_dup_df.apply(lambda x : x if x['Post1'] != x['Post2'] else print("Error: same pair founded", x), axis=1)

# check that the generated pairs are actually non-duplicated
if (pd.merge(dup_df, non_dup_df).shape[0]!=0):
    print("Error: duplicate found")
if (pd.merge(dup_df, non_dup_df.rename(index=str, columns={'Post1':'Post2', 'Post2':'Post1'})).shape[0]!=0):
    print("Error: duplicate found")
    
# add column Duplicate to non_dup_df
non_dup_df = pd.concat([pd.DataFrame(0, index=range(non_dup_df.shape[0]), columns=['Duplicate']), non_dup_df], axis=1)

#clean workspace
del dup_ls
del posts_ls
#%% SPLITTING OF THE DATASET

# shuffling non_dup_df and dup_df
non_dup_df = non_dup_df.sample(frac=1).reset_index(drop=True)
dup_df = dup_df.sample(frac=1).reset_index(drop=True)

# split dataset into train validation and test set
train_index = 12000
validation_index = 500 + train_index

train_df = pd.concat([non_dup_df[0:train_index], dup_df[0:train_index]], ignore_index=True)
validation_df = pd.concat([non_dup_df[train_index:validation_index], dup_df[train_index:validation_index]], ignore_index=True)
test_df = pd.concat([non_dup_df[validation_index:], dup_df[validation_index:]], ignore_index=True)

# check that the sets are disjointed
if (pd.merge(train_df, validation_df).shape[0]!=0):
    print("Error: Train_df and Validation_df are not disjointed")
if (pd.merge(validation_df, test_df).shape[0]!=0):
    print("Error: Validation_df and Test_df are not disjointed")

# shuffling train_df, validation_df and test_df
train_df = train_df.sample(frac=1).reset_index(drop=True)
validation_df = validation_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

#clean workspace
del train_index
del validation_index
del dup_indeces


#%% TOKENIZATION OF QUESTIONS

def question2tokens(q):
    soup = BeautifulSoup(q, 'html5lib')
    #substitute all the links with a unique string
    for link in soup.find_all('a'):
        link.string='thistokenisalink'
    for code in soup.find_all('code'):
        code.string='thistokeniscode'
    #remove all the html tags from the text and make all the words lowercase
    q_text = soup.get_text().lower()
    #define regular expressions for specific text
    time=re.compile(r'([0-1]?\d|2[0-3])(?::([0-5]?\d))+')
    date=re.compile(r'\d\d(\d\d)?(-|/)\d\d(-|/)\d\d(\d\d)?')
    version=re.compile(r'(\d+\.\d+(.\d+)*)|(\S*\d+\S*)+')
    path=re.compile(r'\b(~|\w*|.)?(/\S*)+\b')
    hexadecimal=re.compile(r'\b0?x[0-9a-fA-F]+\b')
    variable=re.compile(r'([a-zA-Z0-9]*_+[a-zA-Z0-9]*)+')
    nochar=re.compile(r'[^a-zA-Z0-9]+')
    #mark the specific text found
    q_text=re.sub(time, 'thistokenistime', q_text)
    q_text=re.sub(date, 'thistokenisadate', q_text)
    q_text=re.sub(version, 'thistokenisaversion', q_text)
    q_text=re.sub(path, 'thistokenisapath', q_text)
    q_text=re.sub(hexadecimal, 'thistokenishexadecimal', q_text)
    q_text=re.sub(variable, 'thistokenisavariable', q_text)
    q_text=re.sub(nochar, ' ', q_text)
    return word_tokenize(q_text)

#tokenize all the questions
qs_tokens = [question2tokens(q) for q in list(posts_df['Text'])]
posts_df['Tokens']=qs_tokens

del qs_tokens

#%% SAVE DATAFRAMES AS CSV FILES

posts_df.to_csv("./posts_df.csv")
test_df.to_csv("./tests_df.csv")
train_df.to_csv("./train_df.csv")
validation_df.to_csv("./validation_df.csv")

#%% WORD EMBEDDINGS: define two helping functions and build the dataset

def emb_build_dataset(posts):
    
    flat_posts = [word for post in posts for word in post]
    #create a list of tuples (word, count) sorted by count
    count = collections.Counter(flat_posts).most_common()
    
    #give a unique index to each word using a dictionary
    dictionary = dict()
    for word, _ in count:
        dictionary[word]=len(dictionary)
    #reverse the dictionary
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    #create a list, where, for each word in the corpus there is the corresponding index
    data = list()
    for word in flat_posts:
        index = dictionary[word]
        data.append(index)
    
    return data, count, dictionary, reversed_dictionary

data_index = 0
# generate batch data
def emb_generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    
    #initilize the buffer with the first span words
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
    #fill batch and context
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    
    return batch, context

#posts_df = pd.read_csv("posts_df.csv")
posts_df = pd.read_csv("posts_df.csv",converters={"Tokens": lambda x: x.strip("[]").split(", ")})

#BUILD THE DATASET
# data: list of the indeces of the words in the text
# dictionary: key=word, value=index
# reversed_dictionary: key=index, value=word
# count: list of tuples of type (word, num_of_occurences_in_the_text)
data, count, dictionary, reversed_dictionary = emb_build_dataset(posts_df['Tokens'])
vocabulary_size = len(dictionary)

#%% WORD EMBEDDINGS: build the skip-gram model with tensorflow

# set the values of hyperparameters of the model
emb_batch_size = 128
skip_window = 1
num_skip = 2
embedding_size = 128
num_sampled = 64

#input layer (note that we don't explicitly need the one-hot style matrix, but only a vector with the indexes of the words)
train_inputs = tf.placeholder(tf.int32, shape=[emb_batch_size])
#output layer (note that we predict )
train_context = tf.placeholder(tf.int32, shape=[emb_batch_size, 1])

#weights between input layer and hidden layer (this will be the matrix of embeddings)
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
#associate each embedding vector with the corresponding word in the input (this is essentially the output of the hidden layer)
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

#weights and biases between hidden and output layer
weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
biases = tf.Variable(tf.zeros([vocabulary_size]))

#define the loss function and the correspondent optimizer
nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights = weights, biases = biases, labels = train_context,inputs = embed, num_sampled = num_sampled, num_classes = vocabulary_size))
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

#VALIDATION OF THE MODEL
#build the validation-set made of the 16 words among the top 200 frequent words
validation_size = 16
validation_window = 200
validation_set = np.random.choice(validation_window, validation_size, replace=False)

#constant to hold the validation set in the tensorflow model
validation_const = tf.constant(validation_set, dtype=tf.int32)

#compute the L2 norm of each embedding
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
#normalize each embedding vector with its L2 norm
normalized_embeddings = embeddings / norm

#take the embeddings of our validation set
validation_embeddings = tf.nn.embedding_lookup(normalized_embeddings, validation_const)

#compute the similarity between each word in the validation-set and each word in the vocabulary
similarity = tf.matmul(validation_embeddings, normalized_embeddings, transpose_b=True)

#%% WORD EMBEDDINGS: perform the training

init = tf.global_variables_initializer()
num_steps = 100001

with tf.Session() as sess:
    init.run()
    
    avg_loss=0
    for step in range(num_steps):
        #generate a batch from the dataset
        batch_inputs, batch_context = emb_generate_batch(data, emb_batch_size, num_skip, skip_window)
        
        #perform the training and update the avg_loss
        _, loss_val = sess.run([optimizer, nce_loss], feed_dict={train_inputs: batch_inputs, train_context: batch_context})
        avg_loss += loss_val
        
        #print average loss every 200 steps
        if step % 2000 == 0:
            if step > 0:
                avg_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', avg_loss)
            avg_loss = 0
        
        # Every 10000 steps print the top-8 similar words to those in the validation set
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(validation_size):
                validation_word = reversed_dictionary[validation_set[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % validation_word
                for k in range(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    
    #get the final result!
    ubuntu_embeddings = normalized_embeddings.eval()

#%% WORD EMBEDDINGS: Plot results
    
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(ubuntu_embeddings[:plot_only, :])
  labels = [reversed_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  
#%% NETWORK ARCHITECTURE

q_length = 0 #TODO: all questions same length
max(posts_df['Tokens'][post_df['Id'].apply(lambda x: len(x))) ##[posts_df['Id'][train_df['Post1']]]
vocabulary_size=100
embedding_size=7

#input layer: [batch_size x 2 x q_length]
x = tf.placeholder(tf.int32, shape=[None, 2, q_length])
y = tf.placeholder(tf.int32, shape=[None])

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

#loss function: mean squared error
loss=tf.losses.mean_squared_error(y, score_layer)

#minimize the loss using gradient descent
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)



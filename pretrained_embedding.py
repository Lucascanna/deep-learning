#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 16:25:22 2018

@author: nicolasosio
"""
import numpy as np
import json

dictionary_wiki = {}
f = open('dlp/data/glove.6B.200d.txt')

with open('dlp/data/dictionary_full.json', 'r') as fp:
    dictionary = json.load(fp)

index = 0
dictionary_wiki['UNK'] = index
index +=1

for line in f:
    values = line.split()
    word = "'" + values[0] + "'"
    coefs = np.asarray(values[1:], dtype='float32')
    dictionary_wiki[word] = coefs
    index += 1
f.close()

result = {}
embedding_matrix = np.zeros((10000, 200))
keys = list(dictionary.keys())[:10000]
for key in keys:
    if key in dictionary_wiki.keys():
        idx = len(result)
        result[key] = idx
        embedding_matrix[idx] = dictionary_wiki[key]

print('Found %s word vectors.' % len(dictionary_wiki))



#%%
with open('dlp/data/dictionary_wiki_10000.json', 'w+') as fp:
    json.dump(result, fp)
            
np.savetxt('dlp/data/embedding_wiki_10000_200.csv', embedding_matrix)


#%%
import json
with open('dictionary_wiki.json', 'r') as fp:
    dictionary_wiki = json.load(fp)

with open('dictionary_full.json', 'r') as fp:
    dictionary = json.load(fp)

words = 0
word = []
for key in dictionary.keys():
    if key in dictionary_wiki.keys():
        words += 1
        word.append(key)
        print("Word founded: ", key)

print("Total words : ", words)

#%%
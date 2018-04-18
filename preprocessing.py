#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk import word_tokenize

#%% READING THE XMLs FILES

#from an xml element containing a post to a dictionary
def parse_post(elem):
    
        post = dict()
        postId = elem.attrib.get('Id')
        post['Id'] = postId
        body = elem.attrib.get('Body')
        post['Body'] = body
        title = elem.attrib.get('Title')
        post['Title'] = title
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
posts_ls = [ parse_post(elem) for elem in root_posts.findall("./row[@PostTypeId='1']")]
posts_df = pd.DataFrame.from_records(posts_ls)

# parsing PostLinks.xml into a pandas dataframe
root_links = ET.parse('PostLinks.xml').getroot()
dup_ls = [ parse_link(elem) for elem in root_links.findall("./row[@LinkTypeId='3']")]
dup_df = pd.DataFrame.from_records(dup_ls)

#%% GENERATION OF NON-DUPLICATED POSTS

# generate a dataframe of random pairs of post indexes
non_dup_df = pd.DataFrame(np.random.randint(0, posts_df.shape[0], size=(dup_df.shape[0],2)), columns=['Post1', 'Post2'])
# substitute indexes with PostIds
non_dup_df = non_dup_df.apply(lambda x : [posts_df.loc[x['Post1'], 'Id'], posts_df.loc[x['Post2'], 'Id']], axis=1)

# check that there are no pairs with the same postId
non_dup_df = non_dup_df.apply(lambda x : x if x['Post1'] != x['Post2'] else print("Error: same pair founded", x), axis=1)

# check that the generated pairs are actually non-duplicated
if (pd.merge(dup_df, non_dup_df).shape[0]!=0):
    print("Error: duplicate found")
if (pd.merge(dup_df, non_dup_df.rename(index=str, columns={'Post1':'Post2', 'Post2':'Post1'})).shape[0]!=0):
    print("Error: duplicate found")
    
# add column Duplicate to non_dup_df
non_dup_df = pd.concat([pd.DataFrame(0, index=range(non_dup_df.shape[0]), columns=['Duplicate']), non_dup_df], axis=1)

#%% SPLITTING OF THE DATASET

# shuffling non_dup_df and dup_df
non_dup_df = non_dup_df.sample(frac=1).reset_index(drop=True)
dup_df = dup_df.sample(frac=1).reset_index(drop=True)

# split dataset into train validation and test set
train_index = int(non_dup_df.shape[0] * 0.8)
validation_index = int(non_dup_df.shape[0] * 0.15) + train_index

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
    
#%% TOKENIZATION OF QUESTIONS

qs = posts_df['Body'][:10]

#remove all the html tags from the text
soups = [BeautifulSoup(q) for q in qs]
qs_text = [soup.get_text() for soup in soups]

#tokenize all the questions
qs_tokens = [word_tokenize(q_text) for q_text in qs_text]
#posts_df['Body']=qs_tokens






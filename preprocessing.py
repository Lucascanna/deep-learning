#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def parse_posts(child):
    
        post = dict()
        postId = child.attrib.get('Id')
        post['Id'] = postId
        body = child.attrib.get('Body')
        post['Body'] = body
        title = child.attrib.get('Title')
        post['Title'] = title
        return post

def parse_links(child):
    link = dict()
    link['Post1'] = child.attrib.get('PostId')
    link['Post2'] = child.attrib.get('RelatedPostId')
    link['Duplicate'] = 1
    return link
    

root_posts = ET.parse('Posts.xml').getroot()
posts = [ parse_posts(child) for child in root_posts.findall("./row[@PostTypeId='1']")]
dataframe_posts = pd.DataFrame.from_records(posts)

root_links = ET.parse('PostLinks.xml').getroot()
links = [ parse_links(child) for child in root_links.findall("./row[@LinkTypeId='3']")]
dataframe_links = pd.DataFrame.from_records(links)

non_dup_df = pd.DataFrame(np.random.randint(0, dataframe_posts.shape[0], size=(dataframe_links.shape[0],2)), columns=['Post1', 'Post2'])
non_dup_df = non_dup_df.apply(lambda x : x if x['Post1'] != x['Post2'] else print("Duplicated found"), axis=1)
merge_df = pd.merge(dataframe_links, non_dup_df)
merge2_df= pd.merge(dataframe_links, non_dup_df.rename(index=str, columns={'Post1':'Post2', 'Post2':'Post1'}))




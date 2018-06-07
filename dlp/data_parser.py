# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:17:33 2018

@author: lucas
"""
import pandas as pd

class XMLtoPandasParser(object):
    """
    Given the XML, load it into a pandas dataframe
    """
    
    def __init__(self):
        super(XMLtoPandasParser, self).__init__()
    
    def parse_post(self, elem):
        """
        From an xml element containing a post to a dictionary
        """
        post = dict()
        postId = elem.attrib.get('Id')
        post['Id'] = postId
        body = elem.attrib.get('Body')
        title = elem.attrib.get('Title')
        post['Text'] = title + body
        return post
    
    
    def parse_link(self, elem):
        """
        From an xml element containing link between two duplicated posts to a dictionary
        """
        link = dict()
        link['Post1Id'] = elem.attrib.get('PostId')
        link['Post2Id'] = elem.attrib.get('RelatedPostId')
        link['isDuplicate'] = 1
        return link
    
    
    def to_dataframe_posts(self, posts_root):
        """
        Index of the Posts dataframe: 'Id'
        Attribute of the Posts dataframe: 'Text' 
        """
        posts_ls = [ self.parse_post(elem) for elem in posts_root]
        posts_df = pd.DataFrame.from_records(posts_ls)
        posts_df.set_index('Id', inplace=True)
        return posts_df
    
    def to_dataframe_links(self, links_root):
        """
        The Links dataframe has default indexes
        Attributes of the Links dataframe are: [Post1Id, Post2Id, IsDuplicate]
        """
        links_ls = [ self.parse_link(elem) for elem in links_root ]
        links_df = pd.DataFrame.from_records(links_ls)
        return links_df
         


    
    
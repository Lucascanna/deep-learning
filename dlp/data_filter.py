# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:07:49 2018

@author: lucas
"""


class XMLDataFilter(object):
    """
    Filters the rows of the xml file that are not useful for our purposes
    """
    
    def __init__(self):
        super(XMLDataFilter, self).__init__()
    
    
    def filter_posts(self, posts_root):
        """
        Filter out posts that are no questions
        """
        return posts_root.findall("./row[@PostTypeId='1']")
        
    
    def filter_links(self, links_root):
        """
        Filter out links that are no duplicates
        """
        return links_root.findall("./row[@LinkTypeId='3']")
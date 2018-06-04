# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:26:08 2018

@author: lucas
"""

import xml.etree.ElementTree as ET

import dlp.util as util

class XMLDataLoader(object):
    """
    The class is used to load data from xml and get the root of the xml_file
    """
    
    def __init__(self):
        super(XMLDataLoader, self).__init__()
        
    
    def load_posts(self):
        """
        Main attributes of the table:
            - Id
            - PostTypeId (for questions PostTypeId=1)
            - Title
            - Body
        """
        
        root_posts = ET.parse(util.POSTS).getroot()
        return root_posts
    
        
    def load_links(self):
        """
        Main attributes of the table:
            - Id
            - PostId
            - RelatedPostId
            - LinkTypeId (for links indicating duplicates LinkTypeId=3)
        """
        
        root_links = ET.parse(util.LINKS).getroot()
        return root_links
    
    
    
        
        


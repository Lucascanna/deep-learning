# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:25:52 2018

@author: lucas
"""

from bs4 import BeautifulSoup

class Tokenizer(object):
    
    def __init__(self, df):
        super(Tokenizer, self).__init__()
        self.soups=[BeautifulSoup(q, 'html5lib') for q in list(df["Text"])]

    
    def replace_links(self):
        for soup in self.soups:
            for link in soup.find_all('a'):
                link.string='thistokenisalink'
                
    
    def replace_code(self):
        for soup in self.soups:
            
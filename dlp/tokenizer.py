# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:25:52 2018

@author: lucas
"""
#%%
from bs4 import BeautifulSoup
from nltk import word_tokenize
import re

import dlp.util as util

class Tokenizer(object):
    
    def __init__(self, df):
        super(Tokenizer, self).__init__()
        self.soups=[BeautifulSoup(q, 'html5lib') for q in list(df["Text"])]

    
    def __replace_links(self, soup):
        """
        Replace the links of a single soup with a unique string
        """
        for link in soup.find_all('a'):
            link.string='thistokenisalink'
        #return soup
    
    
    def __replace_code(self, soup):
        """
        Replace the code of a single soup with a unique string
        """
        for code in soup.find_all('code'):
            code.string='thistokeniscode'
        #return soup
    
    
    def __replace_time(self, soup_text):
        """
        Replace the time format in the text of a single soup with a unique string
        """
        time=re.compile(util.TIME)
        return re.sub(time, 'thistokenistime', soup_text)
    
    
    def __replace_date(self, soup_text):
        """
        Replace the date format in the text of a single soup with a unique string
        """
        date=re.compile(util.DATE)
        return re.sub(date, 'thistokenisdate', soup_text)
    
    
    def __replace_version(self, soup_text):
        """
        Replace the version format in the text of a single soup with a unique string
        """
        version=re.compile(util.VERSION)
        return re.sub(version, 'thistokenisversion', soup_text)
    
    
    def __replace_path(self, soup_text):
        """
        Replace the paths format in the text of a single soup with a unique string
        """
        path=re.compile(util.PATH)
        return re.sub(path, 'thistokenispath', soup_text)
    
    
    def __replace_hexadecimal(self, soup_text):
        """
        Replace the hexadecimal format in the text of a single soup with a unique string
        """
        hexadeciaml=re.compile(util.HEXADECIMAL)
        return re.sub(hexadeciaml, 'thistokenishexadecimal', soup_text)
    
    
    def __replace_variable(self, soup_text):
        """
        Replace the variables format in the text of a single soup with a unique string
        """
        variable=re.compile(util.VARIABLE)
        return re.sub(variable, 'thistokenisvariable', soup_text)
    
    
    def __tokenize_single_soup(self, soup):
        """
        Sanitize and tokenize text of a single soup
        """
        self.__replace_links(soup)
        self.__replace_code(soup)
        soup_text=soup.get_text().lower()
        soup_text=self.__replace_time(soup_text)
        soup_text=self.__replace_date(soup_text)
        soup_text=self.__replace_version(soup_text)
        soup_text=self.__replace_path(soup_text)
        soup_text=self.__replace_hexadecimal(soup_text)
        soup_text=self.__replace_variable(soup_text)
        return word_tokenize(soup_text)
        
        
    def tokenize(self):
        """
        Return the list of sanitized text of all the soups
        """
        return [self.__tokenize_single_soup(soup) for soup in self.soups]
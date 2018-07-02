# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:25:52 2018

@author: lucas
"""
#%%
from bs4 import BeautifulSoup
from nltk import word_tokenize
import re
import spacy
from spacy.tokenizer import Tokenizer 
from spacy.pipeline import Pipe
from selectolax.parser import HTMLParser
from lxml import etree

import dlp.util as util

class TokenizerPosts(object):
    
    def __init__(self, df):
        #super(Tokenizer, self).__init__()
        self.soups=list(df["Text"])

    
#    def __replace_links(self, soup):
#        """
#        Replace the links of a single soup with a unique string
#        """
#        for link in soup.find_all('a'):
#            link.string='thistokenisalink'
#        #return soup
    
    
    def __replace_code(self, soup):
        """
        Replace the code of a single soup with a unique string
        """
        code=re.compile(util.CODE)
        return re.sub(code, 'thistokeniscode', soup)
    
    def __deletenochar(self, soup):
        """
        Delete any no char as \n, \t, .. 
        """
        nochar = re.compile(util.NOCHAR)
        return re.sub(nochar, '', soup)


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
    
    
    def __filter_word(self, token):
        
        token=self.__replace_time(token)
        token=self.__replace_date(token)
        token=self.__replace_version(token)
        token=self.__replace_path(token)
        token=self.__replace_hexadecimal(token)
        token=self.__replace_variable(token)

        return token
    
    def __tokenize_single_soup(self, soup):
        """
        Sanitize and tokenize text of a single soup
        """
        #self.__replace_links(soup)
        soup = self.__replace_code(soup)
        soup = self.__deletenochar(soup)
        root= etree.fromstring(soup, etree.HTMLParser())
        soup_text=(' '.join(root.xpath("//text()")))
#        nlp= spacy.load("en")
#        #nlp.remove_pipe('parser')
#        nlp.remove_pipe('ner')
#        nlp.remove_pipe('tagger')
#        tokenizer = Tokenizer(nlp.vocab)
#        #
#        lemmas = [self.__filter_word(doc.lemma_)  for doc in nlp(soup_text) if not (doc.is_punct or 
#                                      doc.dep_==('det' or 'aux' or 'poss' or 'prep' or 'nsubj' or 'nsubjpass' or 'dobj'))]
#        #lemmas = [(doc, doc.dep_)  for doc in nlp(soup_text) if (doc.is_stop)]
#        return lemmas
        return soup_text
        
        
    def tokenize(self):
        """
        Return the list of sanitized text of all the soups
        """
        docs = [self.__tokenize_single_soup(soup) for soup in self.soups]
        nlp= spacy.load("en")
        #nlp.remove_pipe('parser')
        nlp.remove_pipe('ner')
        nlp.remove_pipe('tagger')
        documents = []
        for d in nlp.pipe(docs):
            lemmas = [self.__filter_word(doc.lemma_) for doc in d if not (doc.is_punct or doc.dep_==('det' or 'aux' or 'poss' or 'prep' or 'nsubj' or 'nsubjpass' or 'dobj'))]
            documents.append(lemmas)
        
        return documents
    
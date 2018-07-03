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
        #self.soups = df
    
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
    
    def __deletetabulation(self, soup):
        """
        Delete any tabulation as \n, \t, .. 
        """
        tab = re.compile(util.TAB)
        return re.sub(tab, '', soup)
    
    def __deletenochar(self, soup):
        """
        Delete any no char as £,$,%,..
        """
        nochar= re.compile(util.NOCHAR)
        if nochar.fullmatch(soup)==None:
            #soup=re.sub(nochar, '', soup)        
            return soup
    


    def __replace_time(self, soup_text):
        """
        Replace the time format in the text of a single soup with a unique string
        """
        time=re.compile(util.TIME)
        if time.fullmatch(soup_text)!=None:
            soup_text=re.sub(time, 'thistokenistime', soup_text)        
        return soup_text
    
    
    def __replace_date(self, soup_text):
        """
        Replace the date format in the text of a single soup with a unique string
        """
        date=re.compile(util.DATE)
        if date.fullmatch(soup_text)!=None:
            soup_text=re.sub(date, 'thistokenisdate', soup_text)        
        return soup_text
    
    
    def __replace_version(self, soup_text):
        """
        Replace the version format in the text of a single soup with a unique string
        """
        version=re.compile(util.VERSION)
        #return re.sub(version, 'thistokenisversion', soup_text)
        if version.fullmatch(soup_text)!=None:
            soup_text=re.sub(version, 'thistokenisversion', soup_text)        
        return soup_text
    
    
    def __replace_path(self, soup_text):
        """
        Replace the paths format in the text of a single soup with a unique string
        """
        path=re.compile(util.PATH)
        if path.fullmatch(soup_text)!=None:
            soup_text=re.sub(path, 'thistokenispath', soup_text)        
        return soup_text
    
    
    def __replace_hexadecimal(self, soup_text):
        """
        Replace the hexadecimal format in the text of a single soup with a unique string
        """
        hexadecimal=re.compile(util.HEXADECIMAL)
        if hexadecimal.fullmatch(soup_text)!=None:
            soup_text= re.sub(hexadecimal, 'thistokenishexadecimal', soup_text)        
        return soup_text
    
    
    def __replace_variable(self, soup_text):
        """
        Replace the variables format in the text of a single soup with a unique string
        """
        variable=re.compile(util.VARIABLE)
        if variable.fullmatch(soup_text)!=None:
            soup_text=re.sub(variable, 'thistokenisvariable', soup_text)        
        return soup_text
    
    
    def __filter_word(self, doc):
        
        lemmas = []
        for token in doc:
            if not (token.is_punct or token.is_space or 
                    token.dep_==('det' or 'aux' or 'poss' or 'prep' or 'nsubj' 
                                 or 'nsubjpass' or 'dobj')):
                if token.like_url:
                    token.lemma_ = 'thistokenisurl'
                token = token.lemma_
                token_tmp = self.__deletenochar(token)
                if token_tmp!=None:
                    token=self.__replace_time(token)
                    token=self.__replace_date(token)
                    token=self.__replace_version(token)
                    token=self.__replace_path(token)
                    token=self.__replace_hexadecimal(token)
                    token=self.__replace_variable(token)
                    lemmas.append(token)
#            print (self.__deletenochar(token))
#        if self.__deletenochar(token)==None:
#            pass
#        else:
#            token=self.__deletenochar(token) 
#            print('ELSE', token)

#        return token
        return lemmas
    
    def __tokenize_single_soup(self, soup):
        """
        Sanitize and tokenize text of a single soup
        """
        #self.__replace_links(soup)
        soup = self.__replace_code(soup)
        soup = self.__deletetabulation(soup)
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
            #lemmas = [self.__filter_word(d) for doc in d if not (doc.is_punct or doc.is_space or doc.dep_==('det' or 'aux' or 'poss' or 'prep' or 'nsubj' or 'nsubjpass' or 'dobj'))]
            lemmas = self.__filter_word(d)
            documents.append(lemmas)
        
        return documents

#posts_df = ['<p>https://127.0.0.1 ciao</p>a  www.ciao.com # & >  tutti /meri/Projects/deep-learning!']
#tokenizer = TokenizerPosts(posts_df)
#result = tokenizer.tokenize()
    
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
===========================================================================================
03 tokenize.py
Topic modeling functions with Latent Dirichlet Allocation
to be called by cluster.py and analyze_cluster.py
 - Qian Yang (qyang1@cs.cmu.edu)
 - References:
 - http://brandonrose.org/clustering
===========================================================================================
"""

import string
import re
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from gensim import corpora, models, similarities
from nltk.tag import pos_tag

# =========== NNP and NNPS strippers =========== #

def strip_proppers(text):
    """strip any proper names from a text
    unfortunately right now this is yanking the first word from a sentence too."""

    # first tokenize by sentence
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    
    # then by word to ensure that punctuation is caught as it's own token
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

def strip_proppers_POS(text):
    """
    #strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
    """
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

# load nltk's English stopwords
stopwords = nltk.corpus.stopwords.words('english')


# =========== tokenizing functions =========== #

def tokenize_it(text):
    """
    tokenizes the text, meaning spliting the texts
    into a list of its respective words, aka tokens.
    """

    # first tokenize by sentence, then by word
    # to ensure that punctuation is caught as it's own token

    tokens = [word.lower() for sent in nltk.sent_tokenize(text)
              for word in nltk.word_tokenize(sent)]

    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    return filtered_tokens


def tokenize_and_stem(text):
    """
    tokenizes and stems each token
    returns the set of stems in the input text.
    """

    filtered_tokens = tokenize_it(text)
    
    # load nltk's SnowballStemmer
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems




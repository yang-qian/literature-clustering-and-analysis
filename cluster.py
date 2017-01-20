#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================
02 cluster.py
clusters the txt files parse.py generates
 - Qian Yang (qyang1@cs.cmu.edu)
 - References:
 - http://brandonrose.org/clustering
=========================================
"""

from __future__ import print_function
# from __future__ imports must occur at the beginning of the file
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs

from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from tabulate import tabulate
from parse import df_to_xml, df_row_to_xml
from tokenize_ import *
from plot_cluster import *
from analyze_cluster import *

####################################== PARAMETERS ==#####################################
# cluster method to use, "mds" or "pca"
method = "mds"

# words whose frequencies will be counted in each cluster
search_terms = ["user experience", "usability", "user modeling", "social", "crowd", "crowd-sourcing", "crowdsourcing", "interactive machine learning", "sensing ","sensor", "intelligent environment", "internet of things", "robotic", "embodiment", "agency", "interface adaptation", "automation", "mixed-initiative", "impaired", "wizard of Oz", "framework", "machine learning framework", "architecture", "AI planning", "cognitive", "sense-making", "recommend", "deep learning", "tag"]

# a pre-determined number of clusters
num_clusters = 7

# the min # of docs that contains the word for a word to be considered.
# i.e. passing 0.2 means the term must be in at least 20% of the document
min_per_df = 0.10

# number of topics to use in topic modeling with LDA
num_topics = 5

####################################== MAIN ==#####################################
# ================ load data ================ #
# df schema: 'query','year','title','author'
# 'url','keyword','venue','abstract'

df_ = pd.read_pickle('pkl/df_before_cluster.pkl')

titles = df_['title']
abstracts = df_['abstract']
years = df_['year']
urls = df_['url']
keywords = df_['keyword']
nos = np.arange(len(df_)) # sequence number

print('<<< Now put %d papers into %d clusters using %s with min_per_df = %s >>>'  %
      (len(df_), num_clusters, method, str(min_per_df)))


# ================ tokenizing ================ #

# apply tokenizing funcs to the list of abstracts
# to create two vocabularies: one stemmed and one only tokenized.

totalvocab_stemmed = []
totalvocab_tokenized = []

# iterate over each abstract

for i in abstracts:

    # for each item in 'abstracts', tokenize/stem

    allwords_stemmed = tokenize_and_stem(i)

    # extend the 'totalvocab_stemmed' list

    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_it(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_paper = pd.DataFrame({'words': totalvocab_tokenized},
                           index=totalvocab_stemmed)

# ==================== tf-idf =================== #
# convert the abstracts list into a tf-idf matrix

tfidf_vectorizer = TfidfVectorizer(max_df = 0.8,
                                   max_features = 200000,
                                   min_df = min_per_df,
                                   stop_words = 'english',
                                   use_idf = True,
                                   tokenizer = tokenize_and_stem,
                                   ngram_range = (1, 3))

# fit the vectorizer to absrtacts

tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)
terms = tfidf_vectorizer.get_feature_names()

# calculate document similarity

dist = 1 - cosine_similarity(tfidf_matrix)

# ================= k-mean cluster ================ #

# run k-mean cluster algorithm

kmeans = KMeans(n_clusters=num_clusters).fit(tfidf_matrix)
clusters = kmeans.labels_.tolist()

# append cluster to main df as a col

df_['cluster'] = clusters

print('number of papers per cluster:')
print(df_['cluster'].value_counts())

# =============== pickle the results ============== #
# pickle/load the model

# uncomment this part to save the model
# joblib.dump(kmeans,  'pkl/doc_cluster.pkl')

# uncomment this part to load the pickle
# kmeans = joblib.load('pkl/doc_cluster.pkl')
# clusters = kmeans.labels_.tolist()

# pickle the df with cluster

df_.to_pickle('pkl/df_after_cluster_%d_%s_%s.pkl' %
              (num_clusters, str(min_per_df)[2:], method))


# ================= cluster analysis ================ #
print ("Now analyzing the clusters...")
analyze_clusters(df_, search_terms, num_topics, num_clusters,
                 'analysis/analyze_%d_%s_%s.txt' % (num_clusters, str(min_per_df)[2:], method))


# ================= transform & plot ================ #

print ("Now transforming and plotting...")
# transform the distance/tfidf matrix to 2D array
if method == "pca":
    plt = pca_(tfidf_matrix, kmeans, num_clusters)
elif method == "mds":
    plt = mds_(dist, kmeans, num_clusters)
plt.savefig('viz/clusters_%d_%s_%s.png' % (num_clusters, str(min_per_df)[2:],
                                           method), dpi=150)
plt.close()

# dendrogram the hierarchical clusters
# and convert it to json file for d3 use

"""# get a list of strings, each starts with paper title and its sequence #
brief = ['{} -- {} -- {}'.format(*t) for t in zip(df_['title'], np.arange(len(df_)), df_['query'])]
# Create dictionary for labeling nodes by their IDs and paper names
id2name = dict(zip(np.arange(len(df_)), brief))

data2D = pd.read_pickle('js/data2D_%s.dat' % method)
# hierachical clustering using data2D that pca_ or mds_ generated
linkage_matrix = dendrogram_(data2D, brief, method) # also plot the dendrogram

# and convert the dendro to json for d3 viz
dendro_to_json(linkage_matrix, 'js/hierarchical_%s' % method, id2name)
"""
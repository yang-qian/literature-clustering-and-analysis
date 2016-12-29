#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
===============================================================================================
04 analyze_cluster.py
analyzes the clusters cluster.py generates
 - Qian Yang (qyang1@cs.cmu.edu)
 - Reference: rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
===============================================================================================
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot
import re
from collections import Counter

from tokenize_ import *
import nltk
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
from nltk.tag import pos_tag

# ====================== variables ====================== #
# empty dict to store cluster analysis result
# to be used by analyze_cluster.pu
global g 
g = {}
"""
global cluster0, cluster1, cluster2, cluster3, cluster4

cluster0 = "=============================\n===== CLUSTER 1 SUMMARY =====\n=============================\n"
cluster1 = "=============================\n===== CLUSTER 2 SUMMARY =====\n=============================\n"
cluster2 = "=============================\n===== CLUSTER 3 SUMMARY =====\n=============================\n"
cluster3 = "=============================\n===== CLUSTER 4 SUMMARY =====\n=============================\n"
cluster4 = "=============================\n===== CLUSTER 5 SUMMARY =====\n=============================\n"
"""
# ================= analysis functions ================= #

def freq_kw(df_):
    global g
    
    # group cluster for aggregation analysis
    grouped = df_["keyword"].groupby(df_['cluster']) 
    
    # get freq kws in each group
    for cluster, group in grouped:
        
        # join all keywords in the cluster into a str
        kws_str = ",".join(filter(None, group))
        
        # extract ACM category code i.e. (H.5.2)
        
        acm_codes = re.findall(r'\(.\.\d.\d*\)', kws_str)
        acm_codes.sort()
        # print (acm_codes)
        
        # remove these codes from kws
        # seperate the kws into a list
        kws_str = re.sub(r'\(.\.\d.\d*\)', '', kws_str).lower()
        kws = re.split(r';|,|\/|\(|\)|:', kws_str)
        
        # remove whitespaces in two ends
        kws = [kw.rstrip().lstrip() for kw in kws]
        
        # count occurances of the non-empty kws
        # save the kw to a sorted tuple list if its occurance > 3
        freq_kws = [(key, value) for key, value in Counter(kws).iteritems()
                    if value > 3 and key]
        freq_kws.sort(key=lambda x: x[1], reverse=True)
        freq_kws = ["== {}, {} times;".format(tup[0], tup[1]) for tup in freq_kws]
        
        # convert to string
        freq_kw_str = "\n".join(map(str, freq_kws[:10]))
        
        # add the kw string to the cluster# 
        targetname = 'cluster{}'.format(cluster)
        g[targetname] += "Top Keywords:\n" + freq_kw_str + "\n-----\n"

def freq_author(df_):
    global g
    
    # group cluster for aggregation analysis
    grouped = df_["author"].groupby(df_['cluster']) 
    
    # get freq authors in each group
    for cluster, group in grouped:
        
        # join all author lists in the cluster into a str
        author_list = reduce(lambda x,y: x+y, filter(None, group))
        
        # count occurances of the author
        # save the name to a sorted tuple list if its occurance > 3
        """freq_authors = ["== {}, {} publications;".format(key, value)
                        for key, value in Counter(author_list).iteritems()
                        if value > 3]
        freq_authors.sort(key=lambda x: x[1], reverse=True)"""
        
        # sort and reformat the list
        freq_authors = [(key.replace("%", ","), value) for key, value in Counter(author_list).iteritems()
                        if value > 3]
        freq_authors.sort(key=lambda x: x[1], reverse=True)
        freq_authors = ["== {}, {} publications;".format(tup[0], tup[1]) for tup in freq_authors]
        
        freq_author_str = "\n".join(map(str, freq_authors[:10]))
        
        # add the author string to the cluster# 
        targetname = 'cluster{}'.format(cluster)
        g[targetname] += "Top Authors:\n" + freq_author_str + "\n-----\n"

        
def histo_yr_by_cluster(df_):
    global g
    
    # earliest year in the df
    min_yr = min(df_["year"])
    
    grouped = df_["year"].groupby(df_['cluster']) 

    bins = np.linspace(1997, 2017, 20)

    for cluster, group in grouped:
        print (group["year"].head())
        pyplot.hist(group["year"].tolist(), bins = bins, alpha=0.5)
        
    pyplot.legend(loc='upper right')
    pyplot.show()



def lda_(text_list, num_topics):
    """
    Latent Dirichlet Allocation implementation with Gensim
    input example: abstracts
    """
    result = ""
    # reference: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
    
    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for raw in text_list:
    
        # clean and tokenize document string
        preprocess = raw.lower()
        tokens = tokenize_it(preprocess)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in stopwords]
        
        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()
        
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word = dictionary, passes=20)
    
    # print the topics as a list of 2-tuples
    topics = ldamodel.show_topics(num_topics = num_topics, num_words = 10, formatted = False)
    
    # reformat to a string of topics
    for topic in topics:
        # iterate over tuples (kw str, float)
        for sub_t in topic[1]:
            result += "== {} * {:.2f} + ".format(sub_t[0], sub_t[1])
        result  += ";\n"
    
    return result
    
    
def freq_topic(df_, num_topics):
    global g
    
    # topic modeling with LDA per cluster
    grouped = df_["abstract"].groupby(df_['cluster']) 

    for cluster, group in grouped:
        freq_topic_str = lda_(group, num_topics)
        
        # add the author string to the cluster# 
        g['cluster{}'.format(cluster)] += "Top Topics:\n" + freq_topic_str + "\n-----\n"

def analyze_clusters(df_, num_topics, num_clusters, fout = 'analysis/analysis/analyze_5_1_mds.txt'):
    # create global var to store analysis results
    global g
    
    for i in np.arange(num_clusters):
        g["cluster{0}".format(i)] = \
        "=============================\n===== CLUSTER %d SUMMARY =====\n=============================\n" % i
    
    freq_kw(df_)
    freq_author(df_)
    freq_topic(df_, num_topics)
    
    with open(fout, "w+") as fout:
    
        # all doc topic modeling with LDA
        fout.write("======================================\n===== TOP TOPICS ACROSS CLUSTERS =====\n======================================\n")
        fout.write(lda_(list(df_['abstract']), 5) + "\n-----\n")
        
        # print per-cluster analysis
        for i in np.arange(num_clusters):
            fout.write(g["cluster{0}".format(i)])
    fout.close()


# ======================== main ======================== #
# uncomment this block to run this script seperately

"""df_ = pd.read_pickle('pkl/df_after_cluster_5_1_mds.pkl')

analyze_clusters(df_, 5) # number of clusters
with open('analysis/analyze_5_1_mds.txt', "w+") as fout:
    
    # all doc topic modeling with LDA
    fout.write("======================================\n===== TOP TOPICS ACROSS CLUSTERS =====\n======================================\n")
    fout.write(lda_(list(df_['abstract']), 5) + "\n-----\n")
    
    # print per-cluster analysis
    fout.write(cluster0)
    fout.write(cluster1)
    fout.write(cluster2)
    fout.write(cluster3)
    fout.write(cluster4)
fout.close()"""



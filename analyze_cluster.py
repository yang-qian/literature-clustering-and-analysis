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
from tabulate import tabulate
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
import re
from collections import Counter

from tokenize_ import *
import nltk
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
from nltk.tag import pos_tag

# ====================== variables ====================== #
# empty dict to store cluster analysis result
global g 
g = {}

# ================= analysis functions ================= #

def kw_cnt(df_, kw):
    """
    given a search term,
    returns an array of ints
    which are occurances of the term in the clusters [0:n]
    """
    
    result = []
    
    # group cluster for aggregation analysis
    grouped = df_["abstract"].groupby(df_['cluster']) 
    
    # get abstracts in each group
    for cluster, group in grouped:
        
        # join all abstracts in the cluster into a str
        abs_str = ",".join(filter(None, group)).lower()
        
        result.extend(["%02d" % abs_str.count(kw)])
        
    return pd.Series(result)


def source_cnt(df_):
    global g
    
    # group cluster for aggregation analysis
    grouped = df_["query"].groupby(df_['cluster']) 
    
    # get query terms in each group
    for cluster, group in grouped:
        
        # join all query term lists in the cluster into a list
        query_list = reduce(lambda x,y: x+y, filter(None, group))
        
        # add the cnt to the cluster# 
        targetname = 'cluster{}'.format(cluster)
        g[targetname] += "Source query counts:\n" + "\n".join(["%s, %d times" % (key, value) for key, value in Counter(query_list).iteritems()]) + "\n-----\n"
        

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
        # save the kw to a sorted tuple list
        
        # uncomment this line to set a freq threthold
        """freq_kws = [(key, value) for key, value in Counter(kws).iteritems()
                    if value > 3 and key]"""
        
        freq_kws = [(key, value) for key, value in Counter(kws).iteritems() if key]
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
        
        # join all author lists in the cluster into a list
        author_list = reduce(lambda x,y: x+y, filter(None, group))
        
        # count occurances of the author and save to a tuple list
        # sort and reformat the list
        
        # uncomment this line to set a freq threthold
        """freq_authors = [(key.replace("%", ","), value) for key, value in Counter(author_list).iteritems()
                        if value > 3]"""
        
        freq_authors = [(key.replace("%", ","), value) for key, value in Counter(author_list).iteritems()]
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
    topics = ldamodel.show_topics(num_topics = num_topics, num_words = 20, formatted = False)
    
    # reformat to a string of topics
    for topic in topics:
        result  += "-- "
        # iterate over tuples (kw str, float)
        for sub_t in topic[1]:
            result += "{} * {:.1f}% + ".format(sub_t[0], sub_t[1] * 100)
        result  += ";\n"
    
    return result
    
    
def freq_topic(df_, num_topics):
    global g
    
    # topic modeling with LDA per cluster
    grouped = df_["abstract"].groupby(df_['cluster']) 

    for cluster, group in grouped:
        freq_topic_str = lda_(group, num_topics)
        
        # add the author string to the cluster# 
        g['cluster{}'.format(cluster)] += "Top Topics:\n" + freq_topic_str + "\n \n"

        
# ================= visualization functions ================= #

def plot_kw_cnt(df_, diemension, kws, yr_range = 1997):
    """
    takes a list of keywords,
    plot their frequencies over the years
    """

    plt.style.use('ggplot')
    
    
    # group cluster for aggregation analysis
    grouped = df_["abstract"].groupby(df_[diemension]) 
    
    for kw in kws:
        result = []
        
        # get abstracts in each group
        for cluster, group in grouped:
            
            if int(cluster) >= yr_range:
                # join all abstracts in the cluster into a str
                abs_str = ",".join(filter(None, group)).lower()

                result.append((int(cluster), abs_str.count(kw)))
        
        plt.plot(zip(*result)[0], zip(*result)[1])
        
    plt.legend(kws, loc='upper left')
    plt.xticks(np.arange(1997,2018, 1))
    plt.show()       


# ===================== main function ===================== #

def analyze_clusters(df_, search_terms, num_topics, num_clusters, \
                     fout = 'analysis/analysis/analyze_5_1_mds.txt'):
    # create global var to store analysis results
    global g
    
    # calculate g[clusters]
    for i in np.arange(num_clusters):
        g["cluster{0}".format(i)] = \
        "=============================\n===== CLUSTER %d SUMMARY =====\n=============================\n" % (i+1)
    
    source_cnt(df_)
    freq_kw(df_)
    freq_author(df_)
    freq_topic(df_, num_topics)
    
    # write into .txt
    with open(fout, "w+") as fout:
        
        fout.write("Number of papers per cluster:\n %s \n-----\n" % str(df_['cluster'].value_counts()))
        
        # all doc topic modeling with LDA
        fout.write("======================================\n===== TOP TOPICS ACROSS CLUSTERS =====\n======================================\n")
        fout.write(lda_(list(df_['abstract']), 5) + "\n-----\n")
        
        # frequcies of identified terms in each cluster
        
        kw_result = PrettyTable()
        kw_result.field_names = ["Keyword", "Occurrences"]
        for query in search_terms:
            kw_result.add_row([query, kw_cnt(df_, query).values])
            kw_result.align = 'c'
        fout.write(kw_result.get_string() + "\n\n")
        
        # print per-cluster analysis
        for i in np.arange(num_clusters):
            fout.write(g["cluster{0}".format(i)])
    fout.close()


# ======================== main ======================== #
# uncomment this block to run this script seperately

pkl_str = "10_1_mds"
search_terms = ["user experience", "usability", "user modeling", "social", "crowd", "crowd-sourcing", "crowdsourcing", "interactive machine learning", "sensing ","sensor", "intelligent environment", "internet of things", "robotic", "embodiment", "agency", "interface adaptation", "automation", "mixed-initiative", "impaired", "wizard of Oz", "framework", "machine learning framework", "architecture", "AI planning", "cognitive", "sense-making", "recommend", "deep learning", "tag"]

df_ = pd.read_pickle('pkl/df_after_cluster_%s.pkl' % pkl_str)
num_topics = 5
num_clusters = int(pkl_str.rsplit('_')[0])

"""analyze_clusters(df_, search_terms, num_topics, num_clusters,
                 'analysis/analyze_%s.txt' % pkl_str)
"""
UX_terms = ["user experience", "usability", "user modeling", "machine learning"]
plot_kw_cnt(df_, 'year', UX_terms)
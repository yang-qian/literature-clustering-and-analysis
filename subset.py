#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
===========================================================
02 subset.py
subsets the pre-clustering dataset for individual analysis
 - Qian Yang (qyang1@cs.cmu.edu)
===========================================================
"""

import pandas as pd
from analyze_cluster import lda_


# ======================== funcs ======================== #

def subset_(df, keyword):
    """
    takes in a keyword
    returns an array of indexes, the papers of which
    contain the keyword in either title, abstract or keywords
    """

    df_ = df[['abstract', 'title', 'keyword']]

    df_.loc[:, 'kw_check'] = df_.apply(lambda row: keyword \
            in ' - '.join(list(row)).lower(), axis=1)

    return list(df_[df_['kw_check'] == True].index.values)


def subset_plus(df, keywords):
    """
    takes in one or more keywords
    returns the matched rows of the df
    """

    if type(keywords) == str:
        return df.loc[subset_(df, keywords), :]
    elif type(keywords) == list:
        concat_indexes = reduce(lambda x, y: x + y, [subset_(df, kw)
                                for kw in keywords])
        concat_indexes.sort()
        return df.loc[list(set(concat_indexes)), :]


def subset_to_csv(df, keywords, fout='N/A'):
    if fout == 'N/A':
        fout = str(keywords)

    subset_plus(df, keywords).to_csv('subsets/%s.csv' % fout)
    
    return subset_plus(df, keywords)

def analyze_subset(df, keywords, num_topics = 10, fout='N/A'):
    subset = subset_plus(df, keywords)
    topics = lda_(subset["abstract"], num_topics)
    
    if fout == 'N/A':
        fout = str(keywords)
    with open("subsets/%s.txt" % fout, "w+") as fout:
        fout.write("Top Topics:\n" + topics + "\n \n")
    fout.close()

# ======================== main ========================= #
# uncomment this block to run this script seperately
df_ = pd.read_pickle('pkl/df_before_cluster.pkl')

subset_to_csv(df_, ['user experience', 'interaction design'], "UX and IxD")
"""
analyze_subset(df_, 'framework')
subset_to_csv(df_, 'policy')

subset_to_csv(df_, ['crowdsourcing', 'crowd-sourcing'], 'crowdsourcing')
subset_to_csv(df_, ['disability', 'impaired'], 'accessibility')
"""
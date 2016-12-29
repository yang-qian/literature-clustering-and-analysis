#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import os
import sys

"""
============================================
01 parse.py
parse .txt search results from www.hcibib.org
to xml for Carrot2 to analyze
 - Qian Yang (qyang1@cs.cmu.edu)
 - References:
 - http://brandonrose.org/clustering
============================================
"""

dir_ = 'search_results/'
test_file = 'refworks_ambient_intelligence_543.txt'

# dict to paper meta data

df_ = pd.DataFrame(columns=('query',
                            'year',
                            'title',
                            'author',
                            'url',
                            'keyword',
                            'venue',
                            'abstract',
                            ))


def parse_dir_to_df(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename.startswith('refworks_'):
            parse_file_to_df(filename)


def parse_file_to_df(file_):
    global df_
    global dir_

    with open('%s/%s' % (dir_, file_), 'r') as f:

        # get the search term that captured this paper

        this_hit = re.search(r"refworks_(.*?)_\d.*.txt", file_).group(1)

        # read .txt search results

        lines = f.readlines()

        # get line number of abstracts

        ab_pos = [i for i in np.arange(len(lines))
                  if lines[i].startswith('AB ')]

        last_i = 0

        # iterate over each paper

        for i in ab_pos:

            # get the abstract

            this_ab = (lines[i])[3:-1]

            this_author = []
            this_url = this_kw = this_vn = this_tt = ''

            # search backwards for other meta data

            for j in np.arange(i, last_i, -1):

                # author

                if lines[j].startswith('A1 '):

                    # replace "," between last and first names with "%"

                    this_author.append((lines[j])[3:-1].replace(',', '%'))
                    
                elif lines[j].startswith('YR '):

                # year

                    this_yr = int((lines[j])[3:7])
                elif lines[j].startswith('UL '):

                # url

                    this_url = (lines[j])[3:-1]
                elif lines[j].startswith('K1 '):

                # keywords

                    this_kw = (lines[j])[3:-1]
                elif lines[j].startswith('T2 '):

                # venue

                    this_vn = (lines[j])[3:-1]
                elif lines[j].startswith('T1 '):

                # title

                    this_tt = (lines[j])[3:-1]

            sys.stdout.write('.')

            # write to df

            df_ = df_.append({
                'query': [this_hit],
                'year': this_yr,
                'title': this_tt,
                'author': this_author,
                'url': this_url,
                'keyword': this_kw,
                'venue': this_vn,
                'abstract': this_ab,
                }, ignore_index=True)

            last_i = i

        print '\nCompleted parsing %s!' % file_


def df_row_to_xml(row):
    xml = ['<item>']
    for field in row.index:
        xml.append('  <field name="{0}">{1}</field>'.format(field,
                   row[field]))
    xml.append('</item>')
    return '\n'.join(xml)


def df_to_xml(df):
    return '\n'.join(df.apply(df_row_to_xml, axis=1))


def drop_dup(dfin):

    # mark up all dup rows

    dfin.loc[:, 'dup'] = dfin.duplicated(['title', 'abstract'],
            keep=False)

    # save a copy of the df w/o dups

    dfout = dfin.drop_duplicates(subset=['title', 'abstract'],
                                 keep=False)

    # pull dup rows out

    dups = dfin[dfin['dup'] == True].groupby('abstract')

    # agg query terms for each set of dups
    # append to the df w/o dups

    for (abstract, other_cols) in dups:
        (this_yr,
         this_tt,
         this_author,
         this_url,
         this_kw,
         this_vn,
         this_ab) = list(other_cols.iloc[0])[1:-1]
        query_list = list(set([row[0] for row in other_cols['query']]))

        dfout = dfout.append({'query': query_list,
                              'year': this_yr,
                              'title': this_tt,
                              'author': this_author,
                              'url': this_url,
                              'keyword': this_kw,
                              'venue': this_vn,
                              'abstract': this_ab,
                              'dup': len(query_list) > 1},
                             ignore_index=True)
    return dfout


####################################== MAIN ==#####################################

if __name__ == '__main__':

    # parse_file_to_df(test_file)

    parse_dir_to_df(dir_)

    # remove duplicates

    df_ = drop_dup(df_)
    
    # pickle the df_
    df_.to_pickle("pkl/df_before_cluster.pkl")

    # save each paper to seperate txt files

    for (index, row) in df_.iterrows():
        with open('data/%d.txt' % index, 'w+') as fout:
            fout.write(row['title'] + '\n')
            fout.write(row['keyword'] + '\n')
            fout.write(row['abstract'])
        fout.close()

    # save to mega xml files

    with open('xml/before_cluster.xml', 'w+') as f1:
        f1.write(df_to_xml(df_))
    f1.close()
    


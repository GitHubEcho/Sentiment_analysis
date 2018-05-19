#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:04:23 2018

@author: haojie
"""

# this file is to define and assign the word vectors for the words in reviews sentences

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
from gensim.models import word2vec

def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')

def buildVecs(filename,model):
    fileVecs = []
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = getWordVecs(wordList,model)
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                #print vecsArray
                #sys.exit()
                fileVecs.append(vecsArray)
    return fileVecs  

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # load word2vec model
    with open("alldata.txt", "r", encoding='utf-8')as ff:
        sen=ff.read().split("\n")
    model=word2vec.Word2Vec(sen, min_count=2, size=50)
#    model.save('word2vec_model')
#     fdir = '/Users/heqian/Desktop/rnn_sentiment/'
    fdir = 'D:/Workplace/Sentiment_analysis'
#    inp = fdir + 'word2vec_model'
#    model = gensim.models.KeyedVectors.load_word2vec_format(inp)
    
#    posInput = buildVecs(fdir + 'pos_cut_stopword.txt',model)
#    negInput = buildVecs(fdir + 'neg_cut_stopword.txt',model)
    posInput = buildVecs('pos_cut_stopword.txt',model)
    negInput = buildVecs('neg_cut_stopword.txt',model)
    # use 1 for positive sentiment， 0 for negative
    Y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))
    X = posInput[:]
    for neg in negInput:
        X.append(neg)
    X = np.array(X)

    # write in file   
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    data = pd.concat([df_y,df_x],axis = 1)
    #print data
    data.to_csv(fdir + 'data.csv')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:36:42 2018

@author: haojie
"""

# get rid of the stopwords ##

import codecs, sys 

def stopWord(sourceFile,targetFile,stopkey):
    sourcef = codecs.open(sourceFile, 'r', encoding='utf-8')
    targetf = codecs.open(targetFile, 'w', encoding='utf-8')
    print('open source file: '+ sourceFile)
    print('open target file: '+ targetFile)
    lineNum = 1
    line = sourcef.readline()
    while line:
#       print('---processing ',lineNum,' article---')
        sentence = delstopword(line,stopkey)
        #print sentence
        targetf.writelines(sentence + '\n')       
        lineNum = lineNum + 1
        line = sourcef.readline()
    print('well done.')
    sourcef.close()
    targetf.close()

def delstopword(line,stopkey):
    wordList = line.split(' ')          
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopkey:
            if word != '\t':
                sentence += word + " "
    return sentence.strip()

if __name__ == '__main__':
    stopkey = [w.strip() for w in codecs.open('stopWord.txt', 'r', encoding='utf-8').readlines()]
    
    sourceFile = 'neg_cut.txt'
    targetFile = 'neg_cut_stopword.txt'
    stopWord(sourceFile,targetFile,stopkey)

    sourceFile = 'pos_cut.txt'
    targetFile = 'pos_cut_stopword.txt'
    stopWord(sourceFile,targetFile,stopkey)
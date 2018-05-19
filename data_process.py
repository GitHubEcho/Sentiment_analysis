#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 18:56:28 2018

@author:
"""

import numpy as np
import tensorflow as tf 
import string 
import jieba 
import jieba.analyse 
import codecs, sys, string, re 


with open("pos.txt","r")as p: 
    pos_reviews=p.read().split("\n")

with open("neg.txt","r")as n:
    neg_reviews=n.read().split("\n")

file1=open("pos1.txt","w")
file2=open("neg1.txt","w")

for i in range(len(pos_reviews)):
    review=pos_reviews[i].split(" ")[-1]
    file1.write(review+"\n")
 
for i in range(len(neg_reviews)):
    review=neg_reviews[i].split(" ")[-1]
    file2.write(review+"\n")


def prepareData(sourceFile,targetFile):
    f = codecs.open(sourceFile, 'r', encoding='utf-8')
    target = codecs.open(targetFile, 'w', encoding='utf-8')
    print('open source file: '+ sourceFile)
    print('open target file: '+ targetFile)

    lineNum = 1
    line = f.readline()
    while line:
        print('---processing ',lineNum,' article---')
        line = clearTxt(line)
        seg_line = sent2word(line)
        target.writelines(seg_line + '\n')       
        lineNum = lineNum + 1
        line = f.readline()
    print('well done.')
    f.close()
    target.close()

# 清洗文本
def clearTxt(line):
    if line != '': 
        line = line.strip()
        intab = ""
        outtab = ""
        trantab = str.maketrans(intab, outtab)
  #      pun_num = string.punctuation + string.digits
   #     line = line.encode('utf-8')
    #    line = line.translate(trantab,pun_num)
     #   line = line.decode("utf8")
        #去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]","",line)
        #去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]","",line) 
    return line

#文本切割
def sent2word(line):
    segList = jieba.cut(line,cut_all=False)    
    segSentence = ''
    for word in segList:
        if word != '\t':
            segSentence += word + " "
    return segSentence.strip()


if __name__ == '__main__':   
    sourceFile = 'neg1.txt'
    targetFile = 'neg_cut.txt'
    prepareData(sourceFile,targetFile)
    
    JMJ = 'pos1.txt'
    targetFile = 'pos_cut.txt'
    prepareData(sourceFile,targetFile)




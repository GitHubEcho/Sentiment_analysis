#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:27:00 2018

@author: haojie
"""
import tensorflow as tf 
import numpy as np 
import pandas as pd 
import random

## import data ##
fdir = 'D:/Workplace/Sentiment_analysis'
df = pd.read_csv(fdir + 'data.csv')
y = df.iloc[:,1]
x = df.iloc[:,2:]

learning_rate = 0.01
training_iters = 100000
batch_size = 64 
display_step = 10 
seq_max_len = 50
n_hidden_units = 32 
n_classes = 2

## make the training and test data ##
train_x = x[1:7000]
train_y = y[1:7000]
test_x = x[7000:-1]
test_y = y[7000:-1]

def deal_with_data(train_x,train_y):
    train_x_revised = train_x.values.tolist()
    train_y_revised1 = train_y.values.tolist()
    train_y_revised=[]
    for i in range(len(train_y_revised1)):
        if train_y_revised1[i]==1.0:
            label=[0.0,1.0]
        else:
            label=[1.0, 0.0]
        train_y_revised.append(label)
    
    
    Data_cat=[]
    for j in range(len(train_x_revised)):
        data=train_x_revised[j]
        label=train_y_revised[j]
        atom=[label,data]
        Data_cat.append(atom)   
    random.shuffle(Data_cat)
    return Data_cat


    
## tf graph input ##
      
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
seqlen = tf.placeholder(tf.int32, [None])

## Define weights ##
#weights = {
#        'in':tf.Variable(tf.random_normal([1,n_hidden_units])),
#        'out':tf.Variable(tf.random_normal([n_hidden_units,1]))}

#biases={
#        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
#        'out':tf.Variable(tf.constant(0.1,shape=[n_classes, ]))}


weights = {
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size)*seq_max_len+(seqlen-1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden_units]),index)
    return tf.matmul(outputs, weights['out'])+biases['out']

## define the loss function
pred = dynamicRNN(x,seqlen, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer= tf.train.GradientDescentOptiizer(learning_rate=learning_rate).minimize(cost)

## classify accuracy ##
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

##initualize ##
init = tf.global_variables_initializer()


batch_id = 0

def getTrainBatch(data, batch_size):
    ## this function is to get the batch of trainging data and its corresponding labels ##
    global batch_id 
    if batch_id == len(data):
        batch_id = 0
        
    batch = data[batch_id:min(batch_id+batch_size,len(data))]
    x_batch = [k[1] for k in batch]
    y_batch = [k[0] for k in batch]
    batch_id = min(batch_id + batch_size, len(data))
    return x_batch, y_batch  

Train_data = deal_with_data(train_x,train_y)
Test_data = deal_with_data(test_x,test_y)
## training ##
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters: 
        batch_x, batch_y = getTrainBatch(Train_data,batch_size)
        batch_seqlen = [50 for i in range(batch_size)]
        sess.run(optimizer, feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen})
        if step%display_step == 0:
            acc = sess.run(accuracy, feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen})
            loss = sess.run(cost, feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen})
            print("Iter"+str(step*batch_size)+", Minibatch Loss=" + \
                  "{:.6f}".format(loss)+",Training Accuracy="+\
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    
    test_data, test_label = getTrainBatch(Test_data, 2204) 
    test_seqlen = [50 for i in range(2204)]
    print("Testing Accuracy:",\
          sess.run(accuracy, feed_dict={x:test_data,y:test_label,seqlen:test_seqlen}))
    
    





















    
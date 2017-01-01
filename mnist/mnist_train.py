#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:29:02 2016

@author: tb

based on https://raw.githubusercontent.com/tensorflow/tensorflow/r0.12/tensorflow/examples/tutorials/mnist/mnist_softmax.py

"""

import math
import numpy as np
import pandas as pd

import tensorflow as tf



#%% read in data

#training_data, training_labels = input('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', reshape_to_image = False)

data = Mnist('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', reshape_to_image = False, labels_to_dummies = True)


#%% split to training and testing datasets

training_data, training_labels = data.get_train_data()
# training_data_sample, training_labels_sample = data.get_random_sample()


#%%


in_dim = training_data.shape[1]
out_dim = training_labels.shape[1]

hidden_layer_1_dim = 20


#%% Define the graph and run



with tf.Graph().as_default():
    
    x = tf.placeholder(tf.float32, [None, training_data.shape[1]])
    
#    with tf.variable_scope('layer'):
#        # Create the model
#        x = tf.placeholder(tf.float32, [None, training_data.shape[1]])
#        W = tf.Variable(tf.zeros([training_data.shape[1], training_labels.shape[1]]))
#        b = tf.Variable(tf.zeros([training_labels.shape[1]]))
#        y = tf.matmul(x, W) + b

    
    with tf.name_scope('hidden_layer_1'):
        w = tf.Variable(tf.truncated_normal([in_dim, hidden_layer_1_dim], 
                                            stddev=1.0 / math.sqrt(float(in_dim))),
                        name='weights')
                        
        b = tf.Variable(tf.zeros([hidden_layer_1_dim]), 
                        name = 'biases')
        
#        hidden1 = tf.nn.relu(tf.matmul(feature_placeholder, w) + b)
        hidden_layer_1 = (tf.matmul(x, w) + b)
    
    # linear output layer
    with tf.name_scope('output_layer'):
        w = tf.Variable(tf.truncated_normal([hidden_layer_1_dim, out_dim], 
                                            stddev=1.0 / math.sqrt(float(hidden_layer_1_dim))),
                        name='weights')
                        
        b = tf.Variable(tf.zeros([out_dim]), 
                        name = 'biases')
        
        y = tf.matmul(hidden_layer_1, w) + b
    
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, training_labels.shape[1]])

    # define goal function and the optimizer
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    train_step = tf.train.GradientDescentOptimizer(.001).minimize(cross_entropy)


    # global variable initializer
    init = tf.global_variables_initializer()
    
    # create a session and run session to initialize variables
    with tf.Session() as sess:
        sess.run(init)
        
        # Train
        tf.global_variables_initializer().run()
        for i in range(1000):
            sess.run(train_step, feed_dict={x: training_data, y_: training_labels})
        
            if (i % 100 == 0) or (i == 999):
                # Test trained model
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
                print("It. %4d" %i,  "   Train accuracy: ", sess.run(accuracy, 
                                                                          feed_dict={x: training_data,
                                                                                     y_: training_labels}),
                                     "   Test accuracy: ", sess.run(accuracy, 
                                                                          feed_dict={x: testing_data,
                                                                                     y_: testing_labels}))


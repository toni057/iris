#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:29:02 2016

@author: tb

based on google's tensorflow example

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
testing_data, testing_labels = data.get_test_data()
# training_data_sample, training_labels_sample = data.get_random_sample()


#%%


in_dim = training_data.shape[1]
out_dim = training_labels.shape[1]

hidden_layer_1_dim = 30
hidden_layer_2_dim = 20

num_iter = 1000


#%%

def create_variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var


#%% simple neural net with flattened features

def simple_nnet(x):
    """
    Build the neural net.
    
    Output is linear.
    """
    # hidden layer 1
    with tf.variable_scope('hidden_layer_1'):
#        w = tf.Variable(tf.truncated_normal([in_dim, hidden_layer_1_dim], 
#                                            stddev=1.0 / math.sqrt(float(in_dim))),
#                        name='weights')
#        b = tf.Variable(tf.zeros([hidden_layer_1_dim]), 
#                        name = 'biases')

        w = create_variable_on_cpu(name='weights', 
                                   shape=[in_dim, hidden_layer_1_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(in_dim)), 
                                   dtype=tf.float32))
        b = create_variable_on_cpu(name='bias',
                                   shape = [hidden_layer_1_dim],
                                   initializer = tf.constant_initializer(0.0))
    
#        hidden_layer_1 = tf.nn.relu(tf.matmul(x, w) + b)
        hidden_layer_1 = tf.nn.sigmoid(tf.matmul(x, w) + b)
#        hidden_layer_1 = (tf.matmul(x, w) + b)


    # hidden layer 2
    with tf.variable_scope('hidden_layer_2'):
        w = create_variable_on_cpu(name='weights', 
                                   shape=[hidden_layer_1_dim, hidden_layer_2_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(hidden_layer_1_dim)), 
                                   dtype=tf.float32))
        b = create_variable_on_cpu(name='bias',
                                   shape = [hidden_layer_2_dim],
                                   initializer = tf.constant_initializer(0.0))
    
        hidden_layer_2 = (tf.matmul(hidden_layer_1, w) + b)

        
    # linear output layer
    with tf.variable_scope('output_layer'):
        w = create_variable_on_cpu(name='weights', 
                                   shape=[hidden_layer_2_dim, out_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(hidden_layer_2_dim)), 
                                   dtype=tf.float32))
        b = create_variable_on_cpu(name='bias',
                                   shape = [out_dim],
                                   initializer = tf.constant_initializer(0.0))
    
        y = (tf.matmul(hidden_layer_2, w) + b)
        
    return y


def loss(y, y_):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    return cross_entropy 

#%% Define the graph and run

with tf.Graph().as_default():
    
    x = tf.placeholder(tf.float32, [None, training_data.shape[1]])
    
    y = simple_nnet(x)
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, training_labels.shape[1]])

    # define goal function and the optimizer
    cross_entropy = loss(y, y_)

    train_step = tf.train.GradientDescentOptimizer(.02).minimize(cross_entropy)
    
    
    # global variable initializer
    init = tf.global_variables_initializer()
    
    # create a session and run session to initialize variables
    with tf.Session() as sess:
        sess.run(init)
        
        # Train
        tf.global_variables_initializer().run()
        for i in range(num_iter):
            training_data_sample, training_labels_sample = data.get_random_sample()
            sess.run(train_step, feed_dict={x: training_data_sample, y_: training_labels_sample})
            
            # print first iteration diagnostics and then every 100 iterations
            if ((i+1) % 100 == 0) or (i == 0):
                # Test trained model
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
                print("It. %4d" % (i+1),  "   Train accuracy: ", sess.run(accuracy, 
                                                                          feed_dict={x: training_data,
                                                                                     y_: training_labels}),
                                     "   Test accuracy: ", sess.run(accuracy, 
                                                                    feed_dict={x: testing_data,
                                                                               y_: testing_labels}))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:29:02 2016

@author: tb

based on google's tensorflow example

"""

import math

import tensorflow as tf



#%% read in data

filename_images = 'train-images.idx3-ubyte'
filename_labels = 'train-labels.idx1-ubyte'

#training_data, training_labels = input('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', reshape_to_image = False)

data = Mnist(filename_images, filename_labels, labels_to_dummies = True)


#%% split to training and testing datasets

training_data, training_labels = data.get_train_data()
testing_data, testing_labels = data.get_test_data()
# training_data_sample, training_labels_sample = data.get_random_sample()


#%%

in_dim = training_data.shape[1]
out_dim = training_labels.shape[1]

hidden_layer_1_dim = 20
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
    # hidden layer 1 (sigmoid)
    with tf.variable_scope('hidden_layer_1'):

        w = create_variable_on_cpu(name='weights', 
                                   shape=[in_dim, hidden_layer_1_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(in_dim)), 
                                   dtype=tf.float32))
        b = create_variable_on_cpu(name='bias',
                                   shape = [hidden_layer_1_dim],
                                   initializer = tf.constant_initializer(0.0))
    
        # hidden_layer_1 = tf.nn.relu(tf.matmul(x, w) + b)
        hidden_layer_1 = tf.nn.sigmoid(tf.matmul(x, w) + b)
        # hidden_layer_1 = (tf.matmul(x, w) + b)


    # hidden layer 2 (linear)
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
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

    

#%% Define the graph and run

with tf.Graph().as_default():
    
    x = tf.placeholder(tf.float32, [None, training_data.shape[1]])
    
    # y holds the neural net calc
    y = simple_nnet(x)
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, training_labels.shape[1]])

    # define goal function and the optimizer
    cross_entropy = loss(y, y_)

    
    # summary snapshot
    tf.summary.scalar(name='loss', tensor=cross_entropy)
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(.02).minimize(cross_entropy)
    
    
    # summary tensor
    summary = tf.summary.merge_all()
    
    
    # create a session and run session to initialize variables
    with tf.Session() as sess:
        
        
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter('/home/tb/Desktop/Data/mnist', sess.graph)

        # global variable initializer
        sess.run(tf.global_variables_initializer())
        
        # Train
        tf.global_variables_initializer().run()
        for i in range(num_iter):
            training_data_sample, training_labels_sample = data.get_random_sample()
            sess.run(optimizer, feed_dict={x: training_data_sample, y_: training_labels_sample})
            
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


#%% convolutional neural network for classifying mnist images

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def cnn(x_image):
    """
    Build the convolutional neural net.
    
    Output is linear.
    """
    
    # first convolution layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolution layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    
    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # readout layer    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    return y_conv, keep_prob


#%%


with tf.Graph().as_default():
    
    training_data, training_labels = data.get_train_data2()
    
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    
    # y holds the neural net calc
    y_conv, keep_prob = cnn(x)
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, training_labels.shape[1]])

    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    
    # create a session and run session to initialize variables
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
    
        for i in range(1000):
            batch = data.get_random_sample2()
            
            if i % 1 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
              
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
            print("test accuracy %g"%accuracy.eval(feed_dict={
                x: data.get_test_data2()[0], y_: data.get_test_data2()[1], keep_prob: 1.0}))

    
    
    
    
    
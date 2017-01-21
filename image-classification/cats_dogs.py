#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:04:35 2017

@author: tb
"""

import os

os.chdir('/home/tb/Desktop/Git/tensorflow/image-classification')


import math
import tensorflow as tf

from mnist_load import Cifar
from layer import Layer

from cats_dogs_load import CatDog


#%%

data = CatDog(dir='/home/tb/Desktop/Data/cats-dogs/', reshape_size=(48,48,3), train_size=0.8)


#%% convolutional neural network for classifying mnist images

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

    
def loss(y, y_):
    """ Loss function
    
    Args:
        y: logits
        y_: labels (each observation is encoded as a vector)
        
    Returns:
        Softmax cross entropy tensor.
    """
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_), name='xentropy_mean')
    
    return cross_entropy


def acc(y, y_):
    """ Accuracy
    
    Args:
        y: logits
        y_: labels
        
    Returns:
        Accuracy - num of correctly classified pieces / total number of obs
    """
    
    # accuracy (num of correctly classified pieces / total number of obs)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return accuracy


class LR():
    """Learning rate class
    
    Args:
        min_lr: minimum learning rate value
        max_lr: maximum learning rate value
        T: time constant
    """
    def __init__(self, min_lr, max_lr, T):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.T = T
    
    def get_lr(self, i):
        """Get the learning rate at iteration i.
        Args:
            i: i-th iteration at which to calculate the learning rate..
        
        Returns:
            Learning rate tensor at iteration i.
        """
        return self.min_lr + (self.max_lr - self.min_lr) * math.exp(-i * self.T)




def cnn(x, NUM_CLASES, keep_prob):
    """Build the convolutional neural net.
    
    Args:
        x: input tensor
        keep_prob: probability to keep the link. P(dropout) = 1-keeo_prob
    """
    
    with tf.variable_scope('conv_layer_1'):
        cl_1 = Layer('conv_layer_1', x).conv([6, 6, C, K], stride=2).batch_norm().activation().get()
    
    with tf.variable_scope('conv_layer_2'):
        cl_2 = Layer('conv_layer_2', cl_1).conv([6, 6, K, L], stride=2).batch_norm().activation().get()
    
    with tf.variable_scope('conv_layer_3'):
        cl_3 = Layer('conv_layer_3', cl_2).conv([3, 3, L, M], stride=2).batch_norm().activation().get()
        cl_3_flattened = tf.reshape(cl_3, shape=[-1, 4 * 4 * M])
        
    with tf.variable_scope('fully_connected_layer_1'):
        hl_1 = Layer('fully_connected_layer_1', cl_3_flattened).linear(4 * 4 * M, N).batch_norm().activation('relu').dropout(0.75).get()

    with tf.variable_scope('output'):
        output = Layer('output', hl_1).linear(N, NUM_CLASES).activation('linear').dropout(0.75).get()
    
    print(x.get_shape())
    print(cl_1.get_shape())
    print(cl_2.get_shape())
    print(cl_3.get_shape())
    print(cl_3_flattened.get_shape())
    print(hl_1.get_shape())
    print(output.get_shape())
    
    return output


#%% settings

K = 6    # first convolutional layer output depth
L = 12   # second convolutional layer output depth
M = 24   # third convolutional layer output depth
N = 100  # fully connected layer

# H: height, W: width, C: channels
H, W, C = (data.H, data.W, data.C)

# number of classes
NUM_CLASES = 2


# learning rate parameters - exponential cooling
MIN_LR = 0.0001
MAX_LR = 0.001
T = 0.0005


#%% create the graph and run the optimisation

with tf.Graph().as_default(), tf.device('/cpu:0'):
    
    # images placeholder
    x = tf.placeholder(dtype=tf.float32, shape=[None, H, W, C], name='x')
    
    # keep probabilites
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    
    # y holds the neural net calc
    y = cnn(x, NUM_CLASES, keep_prob)
    
    # Define loss and optimizer
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASES], name='y_')
    
    # learning rate placeholder
    lr = tf.placeholder(tf.float32, name='lr')
    
    # loss
    cross_entropy = loss(y, y_)
    
    # optimizer
    optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    
    # calculate accuracy
    accuracy = acc(y, y_)

    # learning rate tensor
    learning_rate = LR(MIN_LR, MAX_LR, T)
    
    # add soft placement for sessions (specifically to solve gpu placement issue of tf.nn.moments)
    config = tf.ConfigProto(allow_soft_placement=True)
    
    # create a session and run the optimisation
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())
    
        for i in range(10000):
            batch = data.get_next_batch()
            
            # print accuracy every 200 steps
            if i % 200 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1})
                print("step %4d, training accuracy %.3g"%(i, train_accuracy))
            
#                print("           test accuracy %.3g"%accuracy.eval(feed_dict={
#                    x: data.test_images, y_: data.test_images, keep_prob: 1}))
                
            # run the train step
            optimizer.run(feed_dict={x: batch[0], y_: batch[1], lr: learning_rate.get_lr(i), keep_prob: 0.75})    



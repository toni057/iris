#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 20:29:02 2016

@author: tb

based on https://raw.githubusercontent.com/tensorflow/tensorflow/r0.12/tensorflow/examples/tutorials/mnist/mnist_softmax.py

"""


import numpy as np
import pandas as pd

import tensorflow as tf



#%% read in iris data


csv = 'https://raw.githubusercontent.com/toni057/iris/master/iris.csv'
iris = pd.read_csv(csv)

training_data = iris.iloc[:,:4].as_matrix().astype(np.float32)
training_labels = iris.iloc[:,:4:-1].as_matrix().astype(np.float32)


#%% Define the graph

# Create the model
x = tf.placeholder(tf.float32, [None, training_data.shape[1]])
W = tf.Variable(tf.zeros([training_data.shape[1], training_labels.shape[1]]))
b = tf.Variable(tf.zeros([training_labels.shape[1]]))
y = tf.matmul(x, W) + b


# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, training_labels.shape[1]])


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


#%%

sess = tf.InteractiveSession()

# Train
tf.global_variables_initializer().run()
for _ in range(500):
    sess.run(train_step, feed_dict={x: training_data, y_: training_labels})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Prediction accurary: ", sess.run(accuracy, feed_dict={x: training_data,
                                    y_: training_labels}))

sess.close()




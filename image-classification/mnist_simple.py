"""
Created on Tue Dec  6 20:29:02 2016

@author: tb

based on google's tensorflow example

"""

import os
os.chdir('/home/tb/Desktop/Git/tensorflow/mnist')

import math
import tensorflow as tf

from mnist_load import Mnist


filename_images = '/home/tb/Desktop/Data/mnist/train-images.idx3-ubyte'
filename_labels = '/home/tb/Desktop/Data/mnist/train-labels.idx1-ubyte'


#%% read in data

data = Mnist(filename_images=filename_images, filename_labels=filename_labels, labels_to_dummies = True)


#%% split to training and testing datasets

training_data, training_labels = data.get_train_data(flat=True)
testing_data, testing_labels = data.get_test_data(flat=True)


#%%

in_dim = training_data.shape[1]
out_dim = training_labels.shape[1]

hidden_layer_1_dim = 100
hidden_layer_2_dim = 80
hidden_layer_3_dim = 40
hidden_layer_4_dim = 20
hidden_layer_5_dim = 100

num_iter = 2000


#%%

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


def fully_connecter_layer(name, x, in_dim, out_dim, activation='relu', keep_prob=None):
    """Create fully connected layer.
    Args:
        name: layer name
        x: input tensor
        in_dim: layer input dimension
        out_dim: layer output dimension
        activation: activation function
        keep_prob: probability to keep the neuron connection (1-dropout)
    
    Returns:
        output: layer output tensor
    """
    with tf.variable_scope(name):

        w = create_variable_on_cpu(name='weights', 
                                   shape=[in_dim, out_dim],
                                   initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(in_dim)), 
                                   dtype=tf.float32))
        b = create_variable_on_cpu(name='bias',
                                   shape = [out_dim],
                                   initializer = tf.constant_initializer(0.0))
    
        linear = tf.matmul(x, w) + b

        if activation == 'relu':
            output = tf.nn.relu(linear)
        elif activation == 'sigmoid':
            output = tf.nn.sigmoid(linear)
        elif activation == 'softmax':
            output = tf.nn.softmax(linear)
        elif activation == 'linear':
            output = linear
            
        if keep_prob is not None:
            output = tf.nn.dropout(output, keep_prob)

    return output
    
    

#%% simple neural net with flattened features

def simple_nnet(x):
    """
    Build the neural net.
    
    Output is linear.
    """
    # hidden layer 1
    hl_1 = fully_connecter_layer('hidden_layer_1', x, in_dim, hidden_layer_1_dim, 'relu')

    # hidden layer 2
    hl_2 = fully_connecter_layer('hidden_layer_2', hl_1 , hidden_layer_1_dim, hidden_layer_2_dim, 'relu')
    
    # hidden layer 3
    hl_3 = fully_connecter_layer('hidden_layer_3', hl_2 , hidden_layer_2_dim, hidden_layer_3_dim, 'relu')

    # hidden layer 4
    hl_4 = fully_connecter_layer('hidden_layer_4', hl_3, hidden_layer_3_dim, hidden_layer_4_dim, 'relu')

    # hidden layer 5
    hl_5 = fully_connecter_layer('hidden_layer_5', hl_4, hidden_layer_4_dim, hidden_layer_5_dim, 'linear')

    # output layer 5
    y = fully_connecter_layer('output', hl_5, hidden_layer_5_dim, out_dim, 'linear')

    
    return y



#%% Define the graph and run

with tf.Graph().as_default():
    
    x = tf.placeholder(tf.float32, [None, training_data.shape[1]])
    
    # y holds the neural net calc
    y = simple_nnet(x)
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, training_labels.shape[1]])

    # define goal function and the optimizer
    cross_entropy = loss(y, y_)

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(.005).minimize(cross_entropy)
    # optimizer = tf.train.AdamOptimizer(.01).minimize(cross_entropy)
    
    # create a session and run session to initialize variables
    with tf.Session() as sess:
        
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
                
                print("It. %4d" % (i+1),  
                      "   Train accuracy: ", 
                      sess.run(accuracy, feed_dict={x: training_data, 
                                                    y_: training_labels}),
                      "   Test accuracy: ", 
                      sess.run(accuracy, feed_dict={x: testing_data, 
                                                    y_: testing_labels}))




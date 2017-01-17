"""
Created on Tue Dec  6 20:29:02 2016

@author: tb

based on google's tensorflow example

"""

import os
os.chdir('/home/tb/Desktop/Git/tensorflow/mnist')

import math
import tensorflow as tf

from mnist_load import Cifar
from layer import Layer


filename_images = '/home/tb/Desktop/Data/cifar/cifar-10-batches-py/'
filename_labels = '/home/tb/Desktop/Data/cifar/cifar-10-batches-py/'

#%% read in data

data = Cifar(filename_images=filename_images, labels_to_dummies = True)


#%% split to training and testing datasets

training_data, training_labels = data.get_train_data()
testing_data, testing_labels = data.get_test_data()


#%% settings

K = 6    # first convolutional layer output depth
L = 12   # second convolutional layer output depth
M = 24   # third convolutional layer output depth
N = 100  # fully connected layer

# H: height, W: width, C: channels
H, W, C = training_data.shape[1:]

# number of classes
NUM_CLASES = training_labels.shape[1]


# learning rate parameters - exponential cooling
MIN_LR = 0.0001
MAX_LR = 0.001
T = 0.0005


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
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
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



def conv_layer(name, x, kernel, stride=1, activation='relu', keep_prob=None, max_pool=False):
    """Create a 2d conv layer connected layer.
    
    Args:
        name: name
        kernel: kernel shape
        x: input tensor
        stride: stride
        keep_prob: probability to keep the link. P(dropout) = 1-keeo_prob
    
    Returns:
        Variable tensor, 2d convolution output.
    """
    with tf.variable_scope(name):
        w = create_variable_on_cpu(name='weights', 
                            shape=kernel,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = create_variable_on_cpu(name='bias',
                            shape = kernel[3],
                            initializer = tf.constant_initializer(0.0))
        
        output = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME') + b
        
        mean, variance = tf.nn.moments(output, [0, 1, 2], name='moment')
        output = tf.nn.batch_normalization(output, mean, variance, b, None, 1e-5)

        
        if activation == 'relu':
            output = tf.nn.relu(output)
        elif activation == 'sigmoid':
            output = tf.nn.sigmoid(output)
        elif activation == 'linear':
            pass
        
        if max_pool:
            output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        if keep_prob is not None:
            output = tf.nn.dropout(output, keep_prob)
        
#        print(output.get_shape())
            
    return output


def cnn(x, keep_prob):
    """Build the convolutional neural net.
    
    Args:
        x: input tensor
        keep_prob: probability to keep the link. P(dropout) = 1-keeo_prob
    """
    
    with tf.variable_scope('conv_layer_1'):
        cl_1 = Layer('conv_layer_1', x).conv([6, 6, C, K], stride=1).batch_norm().activation().get()
    
    with tf.variable_scope('conv_layer_2'):
        cl_2 = Layer('conv_layer_2', cl_1).conv([5, 5, K, L], stride=2).batch_norm().activation().get()
    
    with tf.variable_scope('conv_layer_3'):
        cl_3 = Layer('conv_layer_3', cl_2).conv([4, 4, L, M], stride=2).batch_norm().activation().get()
        cl_3_flattened = tf.reshape(cl_3, shape=[-1, 7 * 7 * M])
        
    with tf.variable_scope('fully_connected_layer_1'):
        hl_1 = Layer('fully_connected_layer_1', cl_3_flattened).linear(7 * 7 * M, N).batch_norm().activation('relu').dropout(0.75).get()

    with tf.variable_scope('output'):
        output = Layer('output', hl_1).linear(N, 10).activation('linear').dropout(0.75).get()
    
    return output


#%% create the graph and run the optimisation

with tf.Graph().as_default():
    
    training_data, training_labels = data.get_train_data2()

    with tf.device('/gpu:1'):
        x = tf.placeholder(dtype=tf.float32, shape=[None, H, W, C], name='x')
        
        keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        
        # y holds the neural net calc
        y_conv = cnn(x, keep_prob)
        
        # Define loss and optimizer
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASES], name='y_')
    
        # learning rate placeholder
        lr = tf.placeholder(tf.float32, name='lr')
    
        # loss
        cross_entropy = loss(y_conv, y_)
        
        # optimizer
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        
        # calculate accuracy
        accuracy = acc(y_conv, y_)

    # learning rate tensor
    learning_rate = LR(MIN_LR, MAX_LR, T)
    
    # add soft placement for sessions (specifically to solve gpu placement issue of tf.nn.moments)
    config = tf.ConfigProto(allow_soft_placement=True)
    
    # create a session and run the optimisation
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())
    
        for i in range(10000):
            batch = data.get_random_sample2()
            
            # print accuracy every 200 steps
            if i % 200 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1})
                print("step %4d, training accuracy %.3g"%(i, train_accuracy))
            
                print("           test accuracy %.3g"%accuracy.eval(feed_dict={
                    x: data.get_test_data2()[0], y_: data.get_test_data2()[1], keep_prob: 1}))
                
            # run the train step
            train_step.run(feed_dict={x: batch[0], y_: batch[1], lr: learning_rate.get_lr(i), keep_prob: 0.75})    



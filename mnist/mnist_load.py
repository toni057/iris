#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 20:45:11 2016

@author: tb
"""

#%%

import numpy as np
import struct

import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt



filename_images = 't10k-images.idx3-ubyte'
filename_labels = 't10k-labels.idx1-ubyte'

class Mnist():
    
    def __init__(self, filename_images, filename_labels, reshape_to_image = True, labels_to_dummies = False):
        self.input(filename_images, filename_labels, reshape_to_image)
        self.split_train_test()
        
        if labels_to_dummies:
            self.labels = pd.get_dummies(self.labels).as_matrix().astype(np.float32)
        
    
    def input(self, filename_images, filename_labels, reshape_to_image = True):
        # read images
        with open(filename_images, 'rb') as f:
            MSB_first, N, ROWS, COLS = struct.unpack(">IIII", f.read(16))
            self.images = np.fromfile(f, np.ubyte)
            
            if reshape_to_image == True:
                self.images = self.images.reshape([N, ROWS, COLS, 1])
            else:
                self.images = self.images.reshape([N, ROWS * COLS])
            
        # read labels
        with open(filename_labels, 'rb') as f:
            MSB_first, N = struct.unpack(">II", f.read(8))
            self.labels = np.fromfile(f, np.ubyte)

        self.size = len(self.labels)
        
        return self.images, self.labels


    def split_train_test(self, train_size=0.8):
        self.tr_ind = np.random.rand(len(self.labels)) < train_size
        self.tr_size = sum(self.tr_ind )

        
    def get_random_sample(self, n=100, train_only = True):
        if train_only:
            sample = np.random.randint(0, self.tr_size, n)
            return self.images[self.tr_ind,:][sample,:], self.labels[self.tr_ind,:][sample,:]
        else:
            sample = np.random.randint(0, self.size, n)
            return self.images[sample,:], self.labels[sample,:]

        
    def get_train_data(self, train_only=True):
        if self.tr_ind is None:
            self.split_train_test()
            
        if train_only:
            return self.images[self.tr_ind,:], self.labels[self.tr_ind,:]
        else:
            return self.images, self.labels
            
    
    def get_test_data(self):
        return self.images[~self.tr_ind,:], self.labels[~self.tr_ind,:]
    

def plot(image, revert = True):
    if revert:
        image = 256 - image
    plt.imshow(image, cmap = 'gray')
    plt.show()


data = Mnist(filename_images, filename_labels)
images, labels = (data.images, data.labels)
plot(images[0,:,:,:].reshape([28, 28]))


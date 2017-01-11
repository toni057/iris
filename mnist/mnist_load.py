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

import matplotlib.pyplot as plt

from abc import ABCMeta



class Dataset():
    
    def __init__(self, filename_images, filename_labels='', labels_to_dummies = False):
        self.input(filename_images, filename_labels)
        self.split_train_test()
        
        if labels_to_dummies:
            self.labels = pd.get_dummies(self.labels).as_matrix().astype(np.float32)
        
    def input(self, filename_images, filename_labels):
        pass
        

    def split_train_test(self, train_size=0.8):
        self.tr_ind = np.random.rand(len(self.labels)) < train_size
        self.tr_size = sum(self.tr_ind )

        
    def get_random_sample(self, n=100, train_only = True):
        if train_only:
            sample = np.random.randint(0, self.tr_size, n)
            return self.flat_images[self.tr_ind,:][sample,:], self.labels[self.tr_ind,:][sample,:]
        else:
            sample = np.random.randint(0, self.size, n)
            return self.flat_images[sample,:], self.labels[sample,:]

        
    def get_train_data(self, train_only=True):
        if self.tr_ind is None:
            self.split_train_test()
            
        if train_only:
            return self.flat_images[self.tr_ind,:], self.labels[self.tr_ind,:]
        else:
            return self.flat_images, self.labels
            
    
    def get_test_data(self):
        return self.flat_images[~self.tr_ind,:], self.labels[~self.tr_ind,:]
    
    
    # the *2 version return 2d images
    def get_random_sample2(self, n=100, train_only = True):
        if train_only:
            sample = np.random.randint(0, self.tr_size, n)
            return self.images[self.tr_ind,:,:][sample,:,:], self.labels[self.tr_ind,:][sample,:]
        else:
            sample = np.random.randint(0, self.size, n)
            return self.images[sample,:,:], self.labels[sample,:]

        
    def get_train_data2(self, train_only=True):
        if self.tr_ind is None:
            self.split_train_test()
            
        if train_only:
            return self.images[self.tr_ind,:,:], self.labels[self.tr_ind,:]
        else:
            return self.images, self.labels
            
    
    def get_test_data2(self):
        return self.images[~self.tr_ind,:,:], self.labels[~self.tr_ind,:]


    def plot(self, i, revert = True):
        pass



class Mnist(Dataset):
    
    def input(self, filename_images, filename_labels):
        # read images
        with open(filename_images, 'rb') as f:
            MSB_first, N, ROWS, COLS = struct.unpack(">IIII", f.read(16))
            self.flat_images = np.fromfile(f, np.ubyte)
            
            self.images = self.flat_images.reshape([N, ROWS, COLS, 1])
            self.flat_images = self.images.reshape([N, ROWS * COLS])
            
        # read labels
        with open(filename_labels, 'rb') as f:
            MSB_first, N = struct.unpack(">II", f.read(8))
            self.labels = np.fromfile(f, np.ubyte)

        self.size = len(self.labels)

    
    def plot(self, i, revert = True):
        """
        Plot  i-th image. 
        Revert - should colors be reverted (set to True for white background)
        """
        if revert:
            image = 256 - self.images[i,:,:,:]
        plt.imshow(image.reshape([28, 28]), cmap = 'gray')
        plt.show()


        
class Cifar(Dataset):
    
    ROWS = 32
    COLS = 32
    CHANNELS = 3

    def input(self, filename_images, filename_labels=''):
        # read images
        labels = []
        img = []
        
#        ('./cifar/cifar-10-batches-py/data_batch_%d' %i)
        
        for i in range(1, 2):
            filename_images = './cifar/cifar-10-batches-py/data_batch_%d' %i
            
            with open(filename_images, 'rb') as f:
                for j in range(10000):
                    row = struct.unpack(">3073B", f.read(1 + ROWS * COLS * CHANNELS))
                    if j==789:
                        break
                    labels.append(row[0])
#                    img.append(np.array(row[1:]).reshape([32, 32, 3]))
                    img.append(np.array(row[1:]).reshape([3, 32, 32]).transpose([1, 2, 0]))
            
#            r = np.array(row[1:1025]).reshape([32, 32])
#            g = np.array(row[1025:2049]).reshape([32, 32])
#            b = np.array(row[2049:3073]).reshape([32, 32])
            

        self.labels = np.array(labels, dtype=np.uint8)
        self.images = np.array(img)
            

    def plot(self, i):
        """
        Plot  i-th image. 
        Revert - should colors be reverted (set to True for white background)
        """
        image = self.images[i,:,:,:]
        plt.imshow(image.reshape([32, 32, 3]))
        plt.show()



plt.imshow(np.array(row[1:]).reshape([3, 32, 32]).transpose([1, 2, 0])[:,:,[2, 1, 0]])
plt.show()


data = Cifar('Data/data_batch_1')
data.plot(789)
#data = Mnist('Data/train-images.idx3-ubyte', 'Data/train-labels.idx1-ubyte')
#images, labels = (data.images, data.labels)
#data.plot(12)


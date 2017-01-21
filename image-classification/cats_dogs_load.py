#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:48:56 2017

@author: tb
"""

#%%


import numpy as np

from PIL import Image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import os


class CatDog():
    
    def read_img(self, dir, reshape_size, image_container, label_container=None):

        files = os.listdir(dir)
        
        new_shape = (self.H, self.W)

        for i in range(len(files)):
            
            img = Image.open(dir + files[i])
            img = img.resize(new_shape)
            img = 256-np.array(img.getdata()).reshape([self.H, self.W, self.C])
            
            image_container.append(img)
            if label_container is not None:
                label_container.append(0 if files[i][0] == 'c' else 1)
    

    def __init__(self, dir, reshape_size=(32,32,3), train_size=1):
        self.H, self.W, self.C = reshape_size
        
        self.train_images = []
        self.labels = []
        self.test_images = []
        
        self.read_img(dir + 'train/', reshape_size, self.train_images, self.labels)
        self.read_img(dir + 'test/', reshape_size, self.test_images)
        
        self.split_train_valid(train_size=train_size)
        
        self.train_images=np.array(self.train_images)
        self.test_images=np.array(self.test_images)
        
        self.labels=np.array(self.labels)
        self.labels=np.concatenate([[np.array(self.labels)], [1-np.array(self.labels)]], 0).T
        
        self.batch=0

        
    def split_train_valid(self, train_size=0.8):
        self.tr_ind = np.random.rand(len(self.labels)) < train_size
        self.tr_size = sum(self.tr_ind )
        

    def get_next_batch(self, batch_size=1000):
        
        start = self.batch
        end = min(self.batch + batch_size, sum(self.tr_ind))
        
        if (self.batch + batch_size) >= sum(self.tr_ind):
            self.batch = 0
        else:
            self.batch = end
        
        return (self.train_images[start:end,:,:,:], self.labels[start:end,:])
        
        
        
        
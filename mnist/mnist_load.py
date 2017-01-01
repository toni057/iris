#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 20:45:11 2016

@author: tb
"""

#%%

import numpy as np
import struct

import matplotlib.image as mpimg
import matplotlib.pyplot as plt



filename_images = 't10k-images.idx3-ubyte'
filename_labels = 't10k-labels.idx1-ubyte'


def input(filename_images, filename_labels, reshape_to_image = True):
    # read images
    with open(filename_images, 'rb') as f:
        MSB_first, N, ROWS, COLS = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, np.ubyte)
        
        if reshape_to_image:
            images = images.reshape([N, ROWS, COLS, 1])
        else:
            images = images.reshape([N, ROWS * COLS])
    

    # read labels
    with open(filename_labels, 'rb') as f:
        MSB_first, N = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, np.ubyte)


    return images, labels



def plot(image, revert = True):
    if revert:
        image = 256 - image
    plt.imshow(image, cmap = 'gray')
    plt.show()


images, labels = input(filename_images, filename_labels)
plot(images[0,:,:,:].reshape([28, 28]))


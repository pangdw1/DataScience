# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:09:17 2020

@author: pangdw1
"""

from sklearn import datasets

from sklearn.datasets import get_data_home
get_data_home()

from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

n_samples, n_features = digits.data.shape
print (n_samples, n_features)

print(digits.data[0])
print(digits.target)

print(digits.data.shape)
print (digits.images.shape)

import pylab as plt
fig = plt.figure(figsize=(3, 3))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))
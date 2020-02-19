# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:18:13 2020

@author: pangdw1
"""
import numpy as np

np.zeros(5)
print(np.zeros(5))

a = np.zeros((3,4))


print(a.shape)

print(a.ndim)

print(a.size)

import matplotlib.image as mpimg
from matplotlib import pyplot as plt

imgArray=mpimg.imread('plane.jpg')
plt.imshow(imgArray, interpolation='nearest')
plt.show()

#print(imgArray)
np.zeros((2,3,4))

type(np.zeros((3,4)))

print (np.ones((3,4)))


np.full((3,4), np.pi)
np.array([[1,2,3,4], [10, 20, 30, 40]])
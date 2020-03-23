# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:59:25 2020

@author: -
"""

import numpy as np
import scipy.io
mat = scipy.io.loadmat("./mnist")
X_digits = mat['data'].T
y_digits = mat['label'][0].T
print(mat['label'][0])

print(mat['label'][0])
print("X_digits.shape:", X_digits.shape) #The dimensions of X
print("Unique entries of y_digits:", np.unique(y_digits)) #The classes in y

import pylab as plt
print("Class of first element in our data set: ", y_digits[0])
plt.rc("image", cmap="binary")
#print("Data contained in first row of X:", X_digits[0])
print("Data shape of first row of X: ", X_digits[0].shape)
print("First row of X: " +str(list(X_digits[0])))
print("Transforming the first row of X into a 2 dimensional representation:")
plt.matshow(X_digits[0].reshape(28, 28)) # we reshape the 784 elements row into a 28x28 matrix
ax = plt.gca()
ax.grid(False)

zeros = X_digits[y_digits==0]  # select all the rows of X where y (target value) is zero (i.e. the zero digits)
ones = X_digits[y_digits==1]   # select all the rows of X where y is one (i.e. the one digits)
print("zeros.shape: ", zeros.shape) # print the number of instances of class 0
print("ones.shape: ", ones.shape) # print the number of instances of class 1

y_digits

plt.rc("image", cmap="binary")
plt.matshow(zeros[0].reshape(28, 28)) 
ax = plt.gca()
ax.grid(False)
plt.matshow(ones[0].reshape(28, 28)) 
ax = plt.gca()
ax.grid(False)

X_new = np.vstack([zeros, ones])  # this "stacks" zeros and ones vertically
print("X_new.shape: ", X_new.shape)
y_new = np.hstack([np.repeat(0, zeros.shape[0]), np.repeat(1, ones.shape[0])])
print("y_new.shape: ", y_new.shape)
print("y_new: ", y_new)

from sklearn.utils import shuffle
X_new, y_new = shuffle(X_new, y_new)
X_mnist_train = X_new[:5000]
y_mnist_train = y_new[:5000]
X_mnist_test = X_new[5000:]
y_mnist_test = y_new[5000:]

from sklearn.linear_model import LogisticRegression
logRegModel = LogisticRegression(solver='lbfgs')
logRegModel.fit(X_mnist_train, y_mnist_train)

plt.matshow(logRegModel.coef_.reshape(28, 28))
plt.colorbar()
ax = plt.gca()
ax.grid(False)
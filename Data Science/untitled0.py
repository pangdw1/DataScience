# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:39:18 2020

@author: pangdw1
"""

from sklearn import preprocessing
import numpy as np

np.set_printoptions(suppress=True) #line to prettify the output of numpy arrays so they are not ridiculously long
X = np.array([[ 1000, -1,  0.02],
              [ 1500,  2,  0.07],
              [ 1290,  1.5, -0.03]])
X_scaled = preprocessing.scale(X)
print(X_scaled  )

print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

X_train = X
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled[:10])
print(scaler.mean_)
print(scaler.scale_)

X_test = np.array([[ 1100, -2,  0.03],
              [ 1200,  0.3,  -0.04],
              [ 1050,  1.4, -0.01]])
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled)

import numpy as np
data = np.loadtxt("data/artifical_lin.txt")
data

X = data[:, :-1] # select all the rows [:, in the data object and all the columns except the last one ,:-1
y = data[:, -1] # select all the rows in the last column of the data object
print(X[:10, :])
print(y[:10])
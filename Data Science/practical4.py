# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:54:53 2020

@author: pangdw1
"""
from sklearn import datasets
diabetes = datasets.load_diabetes()

X = diabetes.data
y = diabetes.target
print(X.shape)
print(y.shape)

print(X[0:10,0:3])

print(y[0:10])

from sklearn.utils import shuffle 
X, y = shuffle(X, y, random_state=1)
print(X.shape)
print(y.shape)

print(X.shape)
X = X[:, 2]
print(X.shape)

train_set_size = 250
X_train = X[:train_set_size]
X_test = X[train_set_size:]
print(X_train.shape)
print(X_test.shape)

train_set_size = 250
y_train = y[:train_set_size]
y_test = y[train_set_size:]
print(y_train.shape)
print(y_test.shape)

#%matplotlib inline
import pylab as plt
trainingDataScatterPlot=plt.scatter(X_train, y_train)
testDataScatterPlot=plt.scatter(X_test, y_test)
plt.xlabel('Data')
plt.ylabel('Target')
plt.legend((trainingDataScatterPlot,testDataScatterPlot),("Training data", "Test Data"));

from sklearn import linear_model
linearRegressionModel = linear_model.LinearRegression()
linearRegressionModel.fit(X_train.reshape(-1, 1),y_train);

print(linearRegressionModel.coef_) #theta 1
print(linearRegressionModel.intercept_) #theta 0

import numpy as np
print("Training error: ", np.mean((linearRegressionModel.predict(X_train.reshape(-1, 1)) - y_train) ** 2))
print("Test error: ", np.mean((linearRegressionModel.predict(X_test.reshape(-1, 1)) - y_test) ** 2))

linearRegressionModel.predict(np.array(0.04).reshape(1, -1))

plt.scatter(X_train, y_train, color='blue')

plt.plot(X_train, linearRegressionModel.predict(X_train.reshape(-1, 1)), color='red', linewidth=3);
plt.xlabel('Data')
plt.ylabel('Target')

plt.scatter(X_test, y_test, color='orange')
plt.plot(X_test, linearRegressionModel.predict(X_test.reshape(-1, 1)), color='red', linewidth=3);
plt.xlabel('Data')
plt.ylabel('Target');

print(linearRegressionModel.score(X_test.reshape(-1, 1), y_test))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.random.seed(0)
r=50
xRange=200
X = xRange*np.random.random(size=(r, 1))
noise = 20
yRange=30000
y = yRange - 100*(X.squeeze() + (noise * np.random.randn(r)))
plt.plot(X, y, 'o', color='green');
plt.xlabel('Mileage')
plt.ylabel('Price');

import numpy as np
import matplotlib.pyplot as plt

#if I simply use a normal list data type an error occurs when fitting the linear regression, so I reshape to -1,1
X = np.array([2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,]).reshape(-1,1)
 
y = np.array([3.88,3.95,4.02,4.08,4.13,4.18,4.22,4.26,4.30,4.35,4.38,4.40,4.44, 4.51,4.60,4.63]).reshape(-1,1)
 
plt.plot(X,y,'r*')
plt.xlabel('Year')
plt.ylabel('Population');
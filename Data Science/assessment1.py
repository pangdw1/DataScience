# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:05:33 2020

@author: pangdw1
"""

import pandas as pd
import pylab as plt

data = pd.read_csv('crime.csv', index_col=0)
print(data.head())

import numpy as np
from sklearn.utils import shuffle
feature_cols = ['Education','Police','Income','Inequality']
target = ['Crime']
X = np.array(data[feature_cols])
y = np.array(data[target])
X, y = shuffle(X, y, random_state=1)

#print(X.shape)
## Extract just one column from data
#a = X[:,0] #Take all the rows in X but only the column with index 2
#b = X[:,1]
#c = X[:,2]
#d = X[:,3]
#
#plt.scatter(a, y)
#plt.xlabel('Education')
#plt.ylabel('Crime')
#
#plt.scatter(b, y)
#plt.xlabel('Police')
#plt.ylabel('Crime')
#
#plt.scatter(c, y)
#plt.xlabel('Income')
#plt.ylabel('Crime')
#
#plt.scatter(d, y)
#plt.xlabel('Inequality')
#plt.ylabel('Crime')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5)

from sklearn import linear_model
model = linear_model.LinearRegression()

model.fit(X_train, y_train);

model.intercept_ # theta0

model.coef_ # theta1,  theta2

print(model.predict(np.array([10, 5, 6000, 16]).reshape(1,-1)))
print(model.predict(np.array([8, 11, 4500, 25]).reshape(1,-1)))
print(model.predict(np.array([6, 8, 3780, 17]).reshape(1,-1)))
print(model.predict(np.array([12, 6, 5634, 22]).reshape(1,-1)))


model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X_train, y_train);
print(model.coef_)


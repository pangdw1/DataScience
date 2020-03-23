# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:36:01 2020

@author: -
"""

#%matplotlib inline
import math

theta0=0
theta1=1

def sigmoid(x,t0,t1):
    a = []
    for item in x: 
        a.append(1/(1+math.exp(-(t0+t1*item))))
    return a

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x,theta0,theta1)
plt.plot(x,sig)
plt.show()

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.7,random_state=1)
                  
print("X.shape:", X.shape)
print("y: ", y)

import pylab as plt
plt.prism()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

from sklearn.linear_model import LogisticRegression
logRegModel = LogisticRegression()

X_train = X[:50]
y_train = y[:50]
X_test = X[50:]
y_test = y[50:]

plt.prism()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c='black', marker='^')

logRegModel.fit(X_train, y_train)

print(logRegModel.intercept_) #theta_0
print(logRegModel.coef_) #theta_1 and #theta_2

import numpy as np
def plot_decision_boundary(clf, X):
    w = clf.coef_.ravel()
    a = -w[0] / w[1]
    xx = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]))
    yy = a * xx - clf.intercept_ / w[1]
    plt.plot(xx, yy)
    plt.xticks(())
    plt.yticks(())
y_pred_train = logRegModel.predict(X_train)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train)
plot_decision_boundary(logRegModel, X)

print("Accuracy on training set:", logRegModel.score(X_train, y_train))
y_pred_test = logRegModel.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, marker='^')
#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train)
plot_decision_boundary(logRegModel, X)
print("Accuracy on test set:", logRegModel.score(X_test, y_test))


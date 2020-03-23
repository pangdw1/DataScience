# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def target_function(x, err=0.5):    
    y = 10 - 1. / (x + 0.1)
    if err > 0:
        y = np.random.normal(y, err)
    return y
#%matplotlib inline 
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 1.2, 500)
y = target_function(x, err=0)
plt.plot(x,y)
plt.xlim([-0.2,1.2])

def make_data(N=40, error=1.0, random_seed=1):
    # randomly sample the data
    np.random.seed(random_seed)
    X = np.random.random(N)[:, np.newaxis]
    y = target_function(X.ravel(), error)
    
    return X, y

import numpy as np
import matplotlib.pyplot as plt
X_train, y_train = make_data(40, error=1)
plt.scatter(X_train.ravel(), y_train);
plt.show()

X_test, y_test = make_data(40, error=1, random_seed=4)
plt.scatter(X_test, y_test,c='r');
plt.show()

trainingErrors = []
testErrors = []
modelDegrees = []

from sklearn import metrics
X_plot = np.linspace(-0.1, 1.1, 500)[:, None]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_plot = model.predict(X_plot)

plt.scatter(X_train, y_train)
plt.plot(X_plot, y_plot,c='k')
plt.show()
print("the estimated model coefficients are", model.intercept_, model.coef_)
trainError = metrics.mean_squared_error(model.predict(X_train), y_train)
trainingErrors.append(trainError)
print("mean squared error on training data:", trainError)
testError = metrics.mean_squared_error(model.predict(X_test), y_test)
testErrors.append(testError)
print("mean squared error on test data:", testError)
print("R square on training data:", model.score(X_train,y_train))
print("R square on test data:", model.score(X_test,y_test))
modelDegrees.append(1)

class PolynomialRegression(LinearRegression):
    """Simple Polynomial Regression to 1D data"""
    def __init__(self, degree=1, **kwargs):
        self.degree = degree
        LinearRegression.__init__(self, **kwargs)
        
    def fit(self, X, y):
        if X.shape[1] != 1:
            raise ValueError("Only 1D data valid here")
        Xp = X ** (1 + np.arange(self.degree))
        return LinearRegression.fit(self, Xp, y)
        
    def predict(self, X):
        Xp = X ** (1 + np.arange(self.degree))
        return LinearRegression.predict(self, Xp)
    
degree=2
model = PolynomialRegression(degree)
model.fit(X_train, y_train)
y_plot = model.predict(X_plot)

plt.scatter(X_train, y_train)
plt.plot(X_plot, y_plot)
#plt.plot(X_test, y_test_predictions)
print("the estimated model coefficients are", model.intercept_, model.coef_)
trainError = metrics.mean_squared_error(model.predict(X_train), y_train)
trainingErrors.append(trainError)
print("mean squared error on training data:", trainError)
testError = metrics.mean_squared_error(model.predict(X_test), y_test)
testErrors.append(testError)
print("mean squared error on test data:", testError)
print("R square on training data:", model.score(X_train,y_train))
print("R square on test data:", model.score(X_test,y_test))
modelDegrees.append(degree)

degree=15
model = PolynomialRegression(degree)
model.fit(X_train, y_train)
y_plot = model.predict(X_plot)

plt.scatter(X_train, y_train)
plt.plot(X_plot, y_plot)
plt.ylim(-4, 14)

print("the estimated model coefficients are", model.intercept_, model.coef_)
trainError = metrics.mean_squared_error(model.predict(X_train), y_train)
trainingErrors.append(trainError)
print("mean squared error on training data:", trainError)
testError = metrics.mean_squared_error(model.predict(X_test), y_test)
testErrors.append(testError)
print("mean squared error on test data:", testError)
print("R square on training data:", model.score(X_train,y_train))
print("R square on test data:", model.score(X_test,y_test))
modelDegrees.append(degree)

X2, y2 = make_data(40, error=1,random_seed=2)
model2 = PolynomialRegression(degree=15)
model2.fit(X2, y2)

X2_test = np.linspace(-0.1, 1.1, 500)[:, None]
y2_plot = model2.predict(X_plot)

plt.scatter(X_train, y_train)
plt.plot(X_plot, y_plot)
plt.plot(X_plot, y2_plot, color='red')
plt.ylim(-4, 14)

from sklearn.model_selection import train_test_split

degrees = np.arange(1, 30)

X, y = make_data(100, error=1.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

training_error = []
test_error = []
mse = metrics.mean_squared_error

for d in degrees:
    model = PolynomialRegression(d).fit(X_train, y_train)
    training_error.append(mse(model.predict(X_train), y_train))
    test_error.append(mse(model.predict(X_test), y_test))
    
plt.plot(degrees, training_error, label='training')
plt.plot(degrees, test_error, label='test')
plt.legend()
plt.xlabel('Polynomial Degree (model complexity)')

plt.ylabel('MSE');

import numpy as np
import matplotlib.pyplot as plt
from mlclass import *
import sys
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def normData(x):
    xN = (x - np.mean(x))/(np.max(x)-np.min(x))
    return xN

# load data set
fname = 'data.dat'
fileDType = np.dtype([('interior', np.float), ('exterior', np.float),
                      ('res', np.float), ('hum', np.float), ('pb', np.float)])

data = np.loadtxt(fname, dtype=fileDType)
x1 = np.transpose([data[:]['interior']])
x2 = np.transpose([data[:]['exterior']])
x3 = np.transpose([data[:]['hum']])
x4 = np.transpose([data[:]['pb']])

y = np.transpose([data[:]['res']])

# Normalize data

x1N = normData(x1)
x2N = normData(x2)
x3N = normData(x3)
x4N = normData(x4)

yN = normData(y)

# parameters value (need to automatize)
theta = np.array([[1.0], [1.0], [1.0],[1.0], [1.0]])

# number of training examples
m = len(y)
print('Number of training examples = %i'  % m)

# number of features
n = 4
print('Number of features = %i'  % n)

# inizialization
lamb = 1.0

# Features matrix (including 1's)
X = np.c_[np.ones(m),x1N,x2N,x3N,x4N]
#print(X)

# cost function and gradient
cost = costFunLinReg(X,yN,theta,lamb)
grad =  gradLinReg(X,yN,theta,lamb)
print('Initial cost function = %f' % cost)
#print('Gradient = ', grad)
#print('Initial theta = ', theta)

plt.figure(1)
plt.plot(x1N,yN,'o',label='x1')
plt.plot(x2N,yN,'o',label='x2')
plt.plot(x3N,yN,'o',label='x3')
plt.plot(x4N,yN,'o',label='x4')

#plt.plot(x1,X.dot(theta),'o',label='Initial hypothesis')

# gradient descent

iterations = 10
alpha = 0.01

print('Number of iterations = %i' %iterations)

for j in range(iterations):
    #time.sleep(0.2)
    sys.stdout.write("-")
    sys.stdout.flush()
    for i in range(len(theta)):
        theta[i] = theta[i] - alpha*gradLinReg(X,yN,theta,lamb)[i]


cost = costFunLinReg(X,yN,theta,lamb)
print('Final cost function = %f' % cost)
#print('final theta = ', theta)

#plt.plot(x1,X.dot(theta),'-',label='Hypothesis after %i iterations' % iterations)
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

x1T = 76.8
x2T = 85.4
x3T = 6.3
x4T = 250.

x1N = (x1T - np.mean(x1))/(np.max(x1)-np.min(x1))
x2N = (x2T - np.mean(x2))/(np.max(x2)-np.min(x2))
x3N = (x3T - np.mean(x3))/(np.max(x3)-np.min(x3))
x4N = (x4T - np.mean(x4))/(np.max(x4)-np.min(x4))

yT = 500.
yTN = (yT - np.mean(y))/(np.max(y)-np.min(y))

xTest = np.c_[1.0,x1N,x2N,x3N,x4N]

print(xTest.dot(theta))
print(yTN)

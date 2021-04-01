import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return sigmoid

def costFunLinReg(X,y,theta,lamb):
    m = len(y)
    cost = 1./(2.*m)*(np.transpose(X.dot(theta) - y).dot(X.dot(theta) - y)) + lamb/(2.*m)*sum(sum(theta[1:]*theta[1:]))
    return cost

def gradLinReg(X,y,theta,lamb):
    m = len(y)
    grad = []
    grad0 = 1./m*np.transpose(X[:,[0]]).dot((X.dot(theta)-y))
    
    for i in range(len(theta)-1):
        grad.append(1/m*np.transpose(X[:,[i+1]]).dot((X.dot(theta)-y)) + lamb/m*theta[i+1])

    gradient = np.append(grad0,grad)
    return gradient

def costFuncLogReg(x,y,theta,lamb):
    return costFuncLogReg, gradientLogReg

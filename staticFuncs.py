import numpy as np
import random

alpha = 1.6732632423543772848170429916717
gamma = 1.0507009873554804934193349852946
alphaStar = -1. * alpha * gamma

def sameSign(x, y):
    return ((x >= 0 and y >= 0) or (x < 0 and y < 0))

def directionize(ls):
    if (ls[0] >= ls[1]):
        return 1
    else:
        return -1
        
def softmax(ls):
    sum = 0
    m = max(ls)
    for x in ls:
        sum += np.exp(x - m)
    return [np.exp(i - m) / sum for i in ls]
    
def dSoftmaxdV(ls, i):
    sum = 0
    m = max(ls)
    for x in ls:
        sum += np.exp(x - m)
    s = np.exp(i - m) / sum
    return s * (1 - s)

def softplus(x):
    try:
        return np.log(1 + np.exp(x))
    except:
        return x
    
def dSoftplusdV(x):
    return (1. / (1. + np.exp(-1. * x)))

def SELU(x):
    if x > 0:
        return gamma * x
    else:
        return gamma * alpha * (np.exp(x) - 1.)
    
def dSELUdV(x):
    if x > 0:
        return gamma
    else:
        return gamma * alpha * np.exp(x)
    
def sigmoid(x):
    try:
        return (1. / (1. + np.exp(-1. * x)))
    except:
        if x < 0:
            return 0.
        else:
            return 1.
    
def dSigmoiddV(x):
    try:
        return (1. / (1. + np.exp(-1. * x))) * (1. - (1. / (1. + np.exp(-1. * x))))
    except:
        return 0
        
def dTanhdV(x):
    return 1. / (np.cosh(x) * np.cosh(x))

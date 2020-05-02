import sys
import os
import numpy as np
import random
import time
import params
from dataManager import data
import feedForwardNet
import staticFuncs as sF

class LSTMLayer:
    def __init__(self, layerSize, inVector, zeroethHidden, zeroethState):
        self.inVector = inVector
        self.inputSize = len(inVector)
        self.layerSize = layerSize
        self.zeroethHidden = zeroethHidden
        self.zeroethState = zeroethState
        
        W_f = [0 for x in range (self.inputSize)]
        W_i = [0 for x in range (self.inputSize)]
        W_o = [0 for x in range (self.inputSize)]
        W_c = [0 for x in range (self.inputSize)]
        
        U_f = [[0 for x in range (self.layerSize)] for y in range (self.layerSize)]
        U_i = [[0 for x in range (self.layerSize)] for y in range (self.layerSize)]
        U_o = [[0 for x in range (self.layerSize)] for y in range (self.layerSize)]
        U_c = [[0 for x in range (self.layerSize)] for y in range (self.layerSize)]
        
        b_f = [0 for x in range (self.layerSize)]
        b_i = [0 for x in range (self.layerSize)]
        b_o = [0 for x in range (self.layerSize)]
        b_c = [0 for x in range (self.layerSize)]
        
    def sendThroughCell(self, t, prevHidden, prevState):
        forget = [0 for x in range (self.layerSize)]
        input = [0 for x in range (self.layerSize)]
        output = [0 for x in range (self.layerSize)]
        candidate = [0 for x in range (self.layerSize)]
        state = [0 for x in range (self.layerSize)]
        hidden = [0 for x in range (self.layerSize)]
        
        forget = np.add(np.add(np.dot(W_f, inVector[t]), np.dot(U_f, prevHidden)), b_f)
        forget = [sF.sigmoid(x) for x in forget]
        input = np.add(np.add(np.dot(W_i, inVector[t]), np.dot(U_i, prevHidden)), b_i)
        input = [sF.sigmoid(x) for x in input]
        output = np.add(np.add(np.dot(W_o, inVector[t]), np.dot(U_o, prevHidden)), b_o)
        ouput = [sF.sigmoid(x) for x in forget]
        candidate = np.add(np.add(np.dot(W_c, inVector[t]), np.dot(U_c, prevHidden)), b_c)
        candidate = [np.tanh(x) for x in forget]
        state = np.add(np.multiply(forget, prevState), np.multiply(input, candidate))
        hidden = np.multiply(output, [np.tanh(x) for x in state])
        
        return hidden, state
        
    def sendThroughLayer(self):
        prevHidden = zeroethHidden
        prevState = zeroethState
        for t in range (self.inputSize):
            prevHidden, prevState = self.sendThroughCell(t, prevHidden, prevState)
        return prevHidden
        


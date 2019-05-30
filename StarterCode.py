import numpy as np
import math

class Perceptron:

    def __init__(self, numIn, numOut):
        self.weights = np.random.normal(0, 0.5, size=(numIn, numOut))

    def RELU(self, X):
        return X * (X > 0)

    def dRELU(self, X):
        return np.ceil(X)

    def tanh(self, X):
        self.tanhActivated = np.tanh(X)
        return self.tanhActivated

    def dtanh(self, X):
        return 1 - np.multiply(self.tanhActivated, self.tanhActivated)

class ConvolutionLayer:

    def __init__(self, numIn, numOut, window=2, stride=2):
        self.window = 2
        self.stride = 2
        self.weights = np.random.normal(0, 0.5, size=(numIn, numOut))


########################
### helper functions ###
########################

def logit(X):
    return np.log(X / (1 - X))

def invLogit(X):
    return 1 / (1 + np.exp(-X))
    
def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))


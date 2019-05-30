import numpy as np
import math
from TextHandler import Corpus

class AEPerceptron:

    def __init__(self, numIn, numOut):
        self.weights = np.random.normal(0, 0.5, size=(numIn, numOut))

    def activation(self, X):
        return X * (X > 0)

    def dactivation(self, X):
        return np.ceil(X)

class TransitionPerceptron:

    def __init__(self, numIn, numOut):
        self.weights = np.random.normal(0, 0.5, size=(numIn, numOut))

    def activation(self, X):
        return np.tanh(X)

    def dactivation(self, X):
        return 1 - (np.square(np.tanh(X)))
    

class Autoencoder:

    def __init__(self, features, layerSizes=[100, 50, 25, 50, 100]):
        self.features = features
        self.layers = []
        self.transitions = []
        self.buildLayers(layerSizes)
        self.totalLoss

    def buildLayers(self, layerSizes):
        for l in range(len(layerSizes)):
            if l == 0:
                self.layers.append(AEPerceptron(self.features, layerSizes[0]))
            else:
                self.layers.append(AEPerceptron(layerSizes[l - 1], layerSizes[l]))
                self.transitions.append(TransitionPerceptron(layerSizes[l-1], layerSizes[1]))
        self.layers.append(AEPerceptron(layerSizes[-1], self.features))
        self.transitions.append(TransitionPerceptron(layerSizes[-1], self.features))

    def forwardPass(self, featuresIn):
        h = len(featuresIn)
        for f in range(len(featuresIn)):
            h[f] = []
            for l in range(len(self.layers)):
                if l == 0:
                    h[f].append(np.matmul(featuresIn[f], self.layers[0]))
                h[f].append(np.matmul(
                

    def softMax(self, X):
        exps = [np.exp(i) for i in X]
        sums = sum(exps)
        return [1 / sums for x in X]

    def crossEntropy(self, yHat, Y):
        return -np.log((Y * (Y > 0)) - yHat)

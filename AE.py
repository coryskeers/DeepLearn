import numpy as np
import math
from TextHandler import Corpus

class Layer:

    def __init__(self, numFeatures, numOutputs):
        self.weights = np.random.normal(0, 0.5, size=(numFeatures, numOutputs))
        self.re_weights = np.random.normal(0, 0.5, size=(numOutputs, numFeatures))

    def activation(self, X):
        return X * (X > 0)

    def dactivation(self, X):
        return np.ceil(X)

    def forwardPass(self, X):
        self.out = self.activation(np.matmul(X, self.weights))

    def backwardPass(self, Y, lr):
        backY = self.dactivation(Y)
        backY = np.matmul(backY, np.transpose(self.weights))
        return backY

"""        
class Transition:

    def __init__(self, numOutputs, numFeatures):
        self.weights = np.random.normal(0, 0.5, size=(numOutputs, numFeatures))
        self.sig = 1

    def activation(self, X):
        self.sig = np.where(X >= 0, 1 / (1 + np.exp(-X)), np.exp(X) / (1 + np.exp(X)))
        return self.sig

    def dactivation(self, X):
        return self.sig * (1 - self.sig)

    def forwardPass(self, X):
        self.out = self.activation(np.matmul(X, self.weights))

    def backwardPass(self, Y, lr):
        backY = np.transpose(self.dactivation(Y))
        self.weights = self.weights - lr * backY
        return backY
"""

class Output:

    def __init__(self, numFeatures, numOutputs):
        self.weights = np.random.normal(0, 0.5, size=(numFeatures, numOutputs))

    def softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    def dactivation(self, Y):
        return -Y * (1 - self.out) * self.out

    def forwardPass(self, X):
        self.out = self.softmax(np.matmul(X, self.weights))

    def backwardPass(self, error, Y, lr):
        backY = self.dactivation(Y)
        backY = np.matmul(error * backY, np.transpose(self.weights))
        return backY
    
class Network:

    def __init__(self, lr = 0.05, text='dracula.txt'):
        self.c = Corpus()
        self.lr = lr
        with open(text, encoding="UTF-8") as file:
            s = file.read()
        self.c.addSentences(s)
        self.nextX()

    def nextX(self):
        if len(self.c.phonemeQueue) > 0:
            self.X = self.c.phonemeMatrix()

    def buildLayers(self):
        self.l0 = Layer(len(self.X[0]), 100)
        self.l1 = Layer(100, 50)
        self.l2 = Layer(50, 100)
        self.lout = Output(100, len(self.X[0]))

    def forwardPass(self, FIRST_PASS=False):
        if FIRST_PASS:
            self.l0.forwardPass(self.X)
            self.l1.forwardPass(self.l0.out)
            self.l2.forwardPass(self.l1.out)
            self.lout.forwardPass(self.l2.out)
        else:
            self.l0.forwardPass(self.X)
            self.l1.forwardPass(self.l0.out + self.t1.activation(np.matmul(self.l1.out, self.t1.weights)))
            self.l2.forwardPass(self.l1.out + self.t2.activation(np.matmul(self.l2.out, self.t2.weights)))
            self.lout.forwardPass(self.l2.out)
            
    def crossEntropy(self, Y, yHat, RETURN_ERROR=False):
        #return -np.log((Y * (Y > 0)) - yHat)
        self.error =-Y * np.log(yHat)
        if RETURN_ERROR:
            return np.sum(self.error) / len(self.error)
        else:
            return -1

    def backwardPass(self):
        backY = self.lout.backwardPass(self.error, self.X, self.lr)
        backY = self.l2.backwardPass(backY, self.lr)
        backY = self.l1.backwardPass(backY, self.lr)
        self.l0.backwardPass(backY, self.lr)

    def train(self, n=20):
        for i in range(n):
            if i == 0:
                self.forwardPass(FIRST_PASS=True)
            else:
                self.nextX()
                self.forwardPass()
            if i % 5 == 0:
                print(np.sum(self.crossEntropy(self.X, self.lout.out, RETURN_ERROR=True)))
            else:
                self.crossEntropy(self.X, self.lout.out)
            self.backwardPass()

import numpy as np
import math
from TextHandler import Corpus

class Layer:
    def __init__(self, numFeatures, numOutputs):
        self.weights = np.random.normal(0, 1, size=(numFeatures, numOutputs))

    def activation(self, X):
        return X * (X > 0)

    def delta(self, X):
        return np.ceil(X)

    def forwardPass(self, X):
        self.out = self.activation(np.matmul(X, self.weights))

    def backwardPass(self, out, lr, error):
        delt = self.delta(self.out)
        self.weights = self.weights - (self.weights * delt * lr * error)
        return error

class Output:
    def __init__(self, numFeatures, numOutputs):
        self.weights = np.random.normal(0, 0.5, size=(numFeatures, numOutputs))

    def softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    def delta(self, X):
        return -X * (1 - self.out) * self.out

    def forwardPass(self, X):
        self.out = self.softmax(np.matmul(X, self.weights))

    def backwardPass(self, error, out, lr):
        delt = self.delta(error)
        self.weights = self.weights - (self.weights * delt * lr)
        return error

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

    def forwardPass(self):
        self.l0.forwardPass(self.X)
        self.l1.forwardPass(self.l0.out)
        self.l2.forwardPass(self.l1.out)
        self.lout.forwardPass(self.l2.out)

    def crossEntropy(self, Y, yHat):
        self.error = np.sum(-Y * np.log(yHat)) / len(Y)

    def backwardPass(self):
        Y = self.lout.backwardPass(self.error, self.X, self.lr)
        Y = self.l2.backwardPass(Y, self.lr)
        Y = self.l1.backwardPass(Y, self.lr)
        self.l1.backwardPass(Y, self.lr)

    def train(self, n=20):
        for i in range(n):
            self.forwardPass()
            self.crossEntropy(self.X, self.lout.out)
            if i % 5 == 0:
                print(self.error)
            if i == n-1:
                print(" ".join([self.c.phonemes[x] for x in np.argmax(self.X, axis=1)]))
                print("*********")
                print(" ".join([self.c.phonemes[x] for x in np.argmax(self.lout.out, axis=1)]))
            self.nextX()

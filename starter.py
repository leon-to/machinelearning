import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Performance:
    def __init__(self): #needed to initialize arrays
        self.iterations = []
        self.errorTrain  = []
        self.errorValid = []
        self.errorTest = []
        self.trainSetAcc = []
        self.validSetAcc = []
        self.testSetAcc = []

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x): #implements relu
     return x * (x > 0)

def softmax(x):
    #np.exp(x)/sum(np.exp(x)) #This implementation is not stable https://deepnotes.io/softmax-crossentropy
    # Below is a more stable implementation, from same website
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps) 
    #return np.exp(x)/sum(np.exp(x)) less stable
    
def computeLayer(X, W, b):    
    return np.matmul(W.transpose(), X) + b

def forwardPass(trainData, Wh, Wo, bh, bo):
    Z1 = np.matmul(trainData, Wh)  #sums of hidden layer, in rows
    X1 = relu(Z1 + bh) #outputs of hidden layyer
    Z2 = np.matmul(X1, Wo) #sums of output layer, in rows
    Y =  softmax((Z2 + bo).T).T
    YoneHot = Y
    Yclass = np.argmax(Y, axis = 1) #1 hot
#==============================================================================
#     newY = np.zeros((Y.shape[0], 10))
#     for item in range(0, Y.shape[0]):
#         newY[item][Y[item]] = 1
#==============================================================================

    return YoneHot, Yclass #returns labels
    
def CE(target, prediction): #input should have one-hot targets and predictions as rows
    N = target.shape[0]
    output = np.sum(target*np.log(prediction + 1e-9), axis=1)
    output = -(1/N)*np.sum(output) 
    return output

def gradCE(target, prediction): #return average gradCE
     
    #perform row-wise dot product
    N = target.shape[0]
    output = np.sum(target*np.reciprocal(prediction), axis = 0)
    output = -(1/N)*output #vector of averaged gradients according to the dataset

    return output

# -*- coding: utf-8 -*-
import starter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Load the data
trainData, validData, testData, trainTarget, validTarget, testTarget = starter.loadData()

#shape the data into proper dimensions and initialize some matrices
W = np.zeros((784,1)) #weight matrix
b = np.zeros((3500,1)) #bias matrix
reg = 0 #regularization parameter
trainData = trainData.reshape([3500,784]) #train data matrix
testData = testData.reshape([145,784]) #test data matrix
validData = validData.reshape([100,784]) #validation data matrix

#test data
W = np.matrix('1;1')
b = np.matrix('1;1;1')
trainData = np.matrix('1,1; 2,2; 3,3')
trainTarget = np.matrix('1;0;1')
reg = 0
                              

value = starter.MSE(W, b, trainData, trainTarget, reg) #call MSE that I just implemented, but I need to check it
grad_wrtb, grad_wrtW = starter.gradMSE(W, b, trainData, trainTarget, reg)
# -*- coding: utf-8 -*-
import starter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Load the data
trainData, validData, testData, trainTarget, validTarget, testTarget = starter.loadData()

#shape the data into proper dimensions and initialize some matrices
W = np.zeros((784,1)) #weight matrix
b = np.zeros((1,1)) #bias matrix
trainData = trainData.reshape([3500,784]) #data matrix
testData = testData.reshape([145,784])
meanSquaredError = np.zeros((784,1))


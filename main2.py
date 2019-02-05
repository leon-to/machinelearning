# -*- coding: utf-8 -*-
import starter
import part2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Load the data
trainData, validData, testData, trainTarget, validTarget, testTarget = starter.loadData()

#shape the data into proper dimensions and initialize some matrices
W = np.zeros((784,1)) #weight matrix
b = 0#np.zeros((3500,1)) #bias matrix
reg = 0 #regularization parameter
origTrainData = trainData
origTestData = testData
trainData = trainData.reshape([3500,784]) #train data matrix
testData = testData.reshape([145,784]) #test data matrix
validData = validData.reshape([100,784]) #validation data matrix

alpha = 0.005
epochs = 5000
x = trainData
y = trainTarget
reg = 0.1

# =============================================================================
# 2.2
# dummy, dummy, loss1, acc1 = part2.grad_descent(W, b, trainData, trainTarget, 0.005, epochs, reg, 0.0000001)
# dummy, dummy, loss2, acc2 = part2.grad_descent(W, b, trainData, trainTarget, 0.001, epochs, reg, 0.0000001)
# dummy, dummy, loss3, acc3 = part2.grad_descent(W, b, trainData, trainTarget, 0.0001, epochs, reg, 0.0000001)
# 
# plt.figure(figsize=(15,15))
# plt.ylabel('Loss')
# plt.xlabel('Iterations')
# plt.title('TrainingSet Error of Different Alpha vs. Iterations of Gradient Descent')
# plt.plot(range(len(loss1)), loss1, '-r', label='0.005 (learning rate)')
# plt.plot(range(len(loss2)), loss2, '-g', label='0.001')
# plt.plot(range(len(loss3)), loss3, '-b', label='0.0001')
# plt.legend(loc='upper right')
# =============================================================================


dummy, dummy, loss1, acc1 = part2.grad_descent(W, b, trainData, trainTarget, 0.005, epochs, reg, 0.0000001)
dummy, dummy, loss2, acc2 = part2.grad_descent(W, b, trainData, trainTarget, 0.005, epochs, reg, 0.0000001, 'MSE')
 
plt.figure(figsize=(15,15))
plt.ylabel('Loss/Error')
plt.xlabel('Iterations')
plt.title('Cross Entropy Loss and MSE loss')
plt.plot(range(len(loss1)), loss1, '-r', label='Cross Entropy Loss')
plt.plot(range(len(loss2)), loss2, '-g', label='MSE')
plt.legend(loc='upper right')
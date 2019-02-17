# -*- coding: utf-8 -*-
import starter
import part2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt






def plot_2_2():
    trainData, validData, testData, trainTarget, validTarget, testTarget = part2.loadData()
    #shape the data into proper dimensions and initialize some matrices
    W = np.random.normal(0, 0.5, size = [784,1])
    b = 0#np.zeros((3500,1)) #bias matrix
    reg = 0 #regularization parameter
    trainData = trainData.reshape([3500,784]) #train data matrix
    testData = testData.reshape([145,784]) #test data matrix
    validData = validData.reshape([100,784]) #validation data matrix
    alpha = 0.005
    epochs = 5000
    error_tol = 0.0000001
    lossType = 'CE'
    #_, _, perfRecord = part2.grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, 0.0000001, "CE", validData, validTarget, testData, testTarget )
    
    W,b, perfRecord = starter.grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, \
                                validData, validTarget, testData, testTarget, lossType )
    
    plt.figure(figsize=(15,15))
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('Training/Validation/Test Loss vs. Iterations of Gradient Descent')
    plt.plot(perfRecord.errorTrain, 'r')
    plt.plot(perfRecord.errorValid, 'g')
    plt.plot(perfRecord.errorTest, 'b')
    plt.legend(['Training, fin. loss = %f' % perfRecord.errorTrain[-1] \
                ,'Validation, fin. loss = %f' % perfRecord.errorValid[-1]\
                , 'Test, fin. Loss = %f' % perfRecord.errorTest[-1]], loc = 4)
    
    
    
    plt.figure(figsize=(15,15))
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('Training/Validation/Test Accuracy vs. Iterations')
    plt.plot(perfRecord.trainSetAcc, 'r')
    plt.plot(perfRecord.validSetAcc, 'g')
    plt.plot(perfRecord.testSetAcc, 'b')
    plt.legend(['Training acc = %f' % perfRecord.trainSetAcc[-1] \
                , 'Validation acc = %f' % perfRecord.validSetAcc[-1] \
                , 'Test acc = %f' % perfRecord.testSetAcc[-1]], loc=4)


def plot_2_3():    
#    W = np.random.normal(0, 0.1, size = [784,1])
    #Load the data
    trainData, validData, testData, trainTarget, validTarget, testTarget = starter.loadData()
    
    trainData = trainData.reshape([3500,784]) #train data matrix
    testData = testData.reshape([145,784]) #test data matrix
    validData = validData.reshape([100,784]) #validation data matrix
    
    #weight
    W = np.random.normal(0, 0.5, size = [784,1])
    #bias
    b = 0
    #learning rate
    alpha = 0.005
    #iteration
    epochs = 5000
    #regularization
    reg = 0
    error_tol = 0.0000001
    
    _, _, mse_record = starter.grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, \
                                validData, validTarget, testData, testTarget, 'MSE' )
    _, _, ce_record = starter.grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, \
                                validData, validTarget, testData, testTarget, 'CE')
    
    mse, ce_loss = mse_record.errorTrain, ce_record.errorTrain
    
    plt.figure(figsize=(15,15))
    plt.ylabel('Loss/Error')
    plt.xlabel('Iterations')
    plt.title('Cross Entropy Loss and MSE loss')
    plt.plot(range(len(ce_loss)), ce_loss, '-r', label='Cross Entropy Loss')
    plt.plot(range(len(mse)), mse, '-g', label='MSE')
    plt.legend(loc='upper right')
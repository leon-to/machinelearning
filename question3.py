# -*- coding: utf-8 -*-
import starter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%Load the data
trainData, validData, testData, trainTarget, validTarget, testTarget = starter.loadData()

#shape the data into proper dimensions and initialize some matrices
error_tol = 10**-7
#W = np.zeros((784,1)) #weight matrix
W = np.random.normal(0, 0.1, size = [784,1])
b = 0#np.zeros((3500,1)) #bias matrix
reg = 0#0.1 #regularization parameter
origTrainData = trainData
origTestData = testData
trainData = trainData.reshape([3500,784]) #train data matrix
testData = testData.reshape([145,784]) #test data matrix
validData = validData.reshape([100,784]) #validation data matrix
alpha = 0.001
epochs = 5000 #5000



#%% 3.1.5.2
def plot_3_1_5_2(B1=0.9, B2=0.999, eps=1e-08):
    epochs = 700
    batchSizeParams = [500] 
    
    perfRecordAll = {}
    lossType = 'CE'
    
    W, b, x, y, loss, training_op, reg = starter.buildGraph(B1 = B1, B2 = B2, lossType = lossType, eps = eps)
    #W, b, x, y, loss, training_op, reg = starter.buildGraph(lossType = 'CE', eps=1e-08)
    
    #Question 3, batch gradient descent
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:   
        for k in range(0, len(batchSizeParams)):
            sess.run(init) #make a new graph. 
            
            batch_size = batchSizeParams[k]
            n = len(trainTarget)
            perfRecord = starter.Performance(epochs)
            
            for i in range(epochs):
                #shuffle
                idx =  np.random.permutation(n)
                x_batch, y_batch = trainData[idx], trainTarget[idx]   
                #iteration of minibatch
                for j in range(0, n, batch_size):
                    x_mini, y_mini = x_batch[j:j+batch_size], y_batch[j:j+batch_size]
                    sess.run(training_op, feed_dict = {x: x_mini, y: y_mini})
                    
                W_val, b_val = W.eval(), b.eval()
                
                print('epoch #%i' % i)
                #In the process of implementing saving the performance
                #Save the per iteration errors: Error, Training Set performance, 
                #Validation Set Performance, Test Set performance 
                perfRecord.errorTrain[i] = sess.run(loss, feed_dict = {x: trainData, y: trainTarget})
                perfRecord.errorValid[i] = sess.run(loss, feed_dict = {x: validData, y: validTarget})
                perfRecord.errorTest[i] = sess.run(loss, feed_dict = {x: testData, y: testTarget})
                _, perfRecord.trainSetAcc[i] ,_ = starter.classify(W_val, b_val, trainData, trainTarget, lossType)
                _, perfRecord.validSetAcc[i] ,_ = starter.classify(W_val, b_val, validData, validTarget, lossType)
                _, perfRecord.testSetAcc[i] ,_ = starter.classify(W_val, b_val, testData, testTarget, lossType)
                
            perfRecordAll["{0}".format(batchSizeParams[k])] = perfRecord #save the data     

    #plot
    plt.figure(figsize=(15,15))
#    plt.subplot(2, 1, 1)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('Training/Validation/Test Loss vs. Iterations of Gradient Descent')
    plt.plot(perfRecordAll["500"].errorTrain, 'r')
    plt.plot(perfRecordAll["500"].errorValid, 'g')
    plt.plot(perfRecordAll["500"].errorTest, 'b')
    plt.legend(['Training, fin. loss = %f' % perfRecordAll["500"].errorTrain[-1] \
                ,'Validation, fin. loss = %f' % perfRecordAll["500"].errorValid[-1]\
                , 'Test, fin. Loss = %f' % perfRecordAll["500"].errorTest[-1]], loc = 4)

    #accuracies
    plt.figure(figsize=(15,15))
#    plt.subplot(2, 1, 2)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('Training/Validation/Test Accuracy vs. Iterations')
    plt.plot(perfRecordAll["500"].trainSetAcc, 'r')
    plt.plot(perfRecordAll["500"].validSetAcc, 'g')
    plt.plot(perfRecordAll["500"].testSetAcc, 'b')
    plt.legend(['Training acc = %f' % perfRecordAll["500"].trainSetAcc[-1] \
                , 'Validation acc = %f' % perfRecordAll["500"].validSetAcc[-1] \
                , 'Test acc = %f' % perfRecordAll["500"].testSetAcc[-1]], loc=4)
    
#%% 3.1.5.3
def plot_3_1_5_3():
    epochs = 700
    batchSizeParams = [100, 700, 1750] #[500] #[100, 700, 1750]
    
    perfRecordAll = {}
    lossType = 'CE'
    
    W, b, x, y, loss, training_op, reg = starter.buildGraph(B1 = 0.9, B2 = 0.999, lossType = lossType, eps = 1e-08)
    #W, b, x, y, loss, training_op, reg = starter.buildGraph(lossType = 'CE', eps=1e-08)
    
    #Question 3, batch gradient descent
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:   
        for k in range(0, len(batchSizeParams)):
            sess.run(init) #make a new graph. 
            
            batch_size = batchSizeParams[k]
            n = len(trainTarget)
            perfRecord = starter.Performance(epochs)
            
            for i in range(epochs):
                #shuffle
                idx =  np.random.permutation(n)
                x_batch, y_batch = trainData[idx], trainTarget[idx]   
                #iteration of minibatch
                for j in range(0, n, batch_size):
                    x_mini, y_mini = x_batch[j:j+batch_size], y_batch[j:j+batch_size]
                    sess.run(training_op, feed_dict = {x: x_mini, y: y_mini})
                    
                W_val, b_val = W.eval(), b.eval()
                
                print('epoch #%i' % i)
                #In the process of implementing saving the performance
                #Save the per iteration errors: Error, Training Set performance, 
                #Validation Set Performance, Test Set performance 
                perfRecord.errorTrain[i] = sess.run(loss, feed_dict = {x: trainData, y: trainTarget})
                perfRecord.errorValid[i] = sess.run(loss, feed_dict = {x: validData, y: validTarget})
                perfRecord.errorTest[i] = sess.run(loss, feed_dict = {x: testData, y: testTarget})
                _, perfRecord.trainSetAcc[i] ,_ = starter.classify(W_val, b_val, trainData, trainTarget, lossType)
                _, perfRecord.validSetAcc[i] ,_ = starter.classify(W_val, b_val, validData, validTarget, lossType)
                _, perfRecord.testSetAcc[i] ,_ = starter.classify(W_val, b_val, testData, testTarget, lossType)
                
            perfRecordAll["{0}".format(batchSizeParams[k])] = perfRecord #save the data     
                
   

    plt.figure(figsize=(15,15))
    plt.subplot(2, 2, 1)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('TrainingSet Error of Different Batch Size vs. Iterations of Gradient Descent')
    plt.plot(perfRecordAll["100"].errorTrain, 'r')
    plt.plot(perfRecordAll["700"].errorTrain, 'b') 
    plt.plot(perfRecordAll["1750"].errorTrain, 'g') 
    plt.legend(['Batch Size 100, fin. loss = %f' % perfRecordAll["100"].errorTrain[-1] \
                ,'Batch Size 700 fin. loss = %f' % perfRecordAll["700"].errorTrain[-1]\
                , 'Batch Size 1750 fin. loss = %f' % perfRecordAll["1750"].errorTrain[-1]])
    
    plt.subplot(2, 2, 2)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('ValidSet Error of Different Batch Size vs. Iterations of Gradient Descent')
    plt.plot(perfRecordAll["100"].errorValid, 'r')
    plt.plot(perfRecordAll["700"].errorValid, 'b') 
    plt.plot(perfRecordAll["1750"].errorValid, 'g') 
    plt.legend(['Batch Size 100, fin. loss = %f' % perfRecordAll["100"].errorValid[-1] \
                ,'Batch Size 700 fin. loss = %f' % perfRecordAll["700"].errorValid[-1]\
                , 'Batch Size 1750 fin. loss = %f' % perfRecordAll["1750"].errorValid[-1]])
    
    plt.subplot(2, 2, 3)
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('TestSet Error of Different Batch Size vs. Iterations of Gradient Descent')
    plt.plot(perfRecordAll["100"].errorTest, 'r')
    plt.plot(perfRecordAll["700"].errorTest, 'b') 
    plt.plot(perfRecordAll["1750"].errorTest, 'g') 
    plt.legend(['Batch Size 100, fin. loss = %f' % perfRecordAll["100"].errorTest[-1] \
                ,'Batch Size 700 fin. loss = %f' % perfRecordAll["700"].errorTest[-1]\
                 , 'Batch Size 1750 fin. loss = %f' % perfRecordAll["1750"].errorTest[-1]])
    
    
    #accuracies
    plt.figure(figsize=(15,15))
    plt.subplot(2, 2, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('TrainingSet Accuracy of Different Batch Size vs. Iterations')
    plt.plot(perfRecordAll["100"].trainSetAcc, 'r')
    plt.plot(perfRecordAll["700"].trainSetAcc, 'b') 
    plt.plot(perfRecordAll["1750"].trainSetAcc, 'g') 
    plt.legend(['Batch Size 100, fin. acc = %f' % perfRecordAll["100"].trainSetAcc[-1] \
                ,'Batch Size 700 fin. acc = %f' % perfRecordAll["700"].trainSetAcc[-1]\
                , 'Batch Size 1750 fin. acc = %f' % perfRecordAll["1750"].trainSetAcc[-1]], loc = 4)
    
    plt.subplot(2, 2, 2)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('ValidSet Accuracy of Different Batch Size vs. Iterations')
    plt.plot(perfRecordAll["100"].validSetAcc, 'r')
    plt.plot(perfRecordAll["700"].validSetAcc, 'b') 
    plt.plot(perfRecordAll["1750"].validSetAcc, 'g') 
    plt.legend(['Batch Size 100, fin. acc = %f' % perfRecordAll["100"].validSetAcc[-1] \
                ,'Batch Size 700 fin. acc = %f' % perfRecordAll["700"].validSetAcc[-1]\
                , 'Batch Size 1750 fin. acc = %f' % perfRecordAll["1750"].validSetAcc[-1]], loc = 4)
    
    plt.subplot(2, 2, 3)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('TestSet Accuracy of Different Batch Size vs. Iterations o')
    plt.plot(perfRecordAll["100"].testSetAcc, 'r')
    plt.plot(perfRecordAll["700"].testSetAcc, 'b') 
    plt.plot(perfRecordAll["1750"].testSetAcc, 'g') 
    plt.legend(['Batch Size 100, fin. acc = %f' % perfRecordAll["100"].testSetAcc[-1] \
                ,'Batch Size 700 fin. acc = %f' % perfRecordAll["700"].testSetAcc[-1]\
                , 'Batch Size 1750 fin. acc = %f' % perfRecordAll["1750"].testSetAcc[-1]], loc = 4)


#%% 3.1.5.4
def plot_3_1_5_4(B1=0.9, B2=0.999, eps=1e-08):
    plot_3_1_5_2(B1, B2, eps)

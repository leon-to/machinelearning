# -*- coding: utf-8 -*-

import starter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time



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
alpha = 0.005
epochs = 5000 #5000


#%% Q4c, MSE

epochs = 700
batch_size = 500
epsParams = [1e-09, 1e-04]
B1=0.9; B2 = 0.999;

perfRecordAll = {}
    
for k in range(0, len(epsParams)):
    eps = epsParams[k]
    W, b, y_pred, x, y, loss, training_op, reg = starter.buildGraph(lossType = 'MSE', B1 = B1, B2 = B2, eps = eps)
    #Question 3, batch gradient descent
    init = tf.global_variables_initializer()
    with tf.Session() as sess:   
        sess.run(init) #make a new graph. 
        
        n = len(trainTarget)
        n_batches = int(n/batch_size)
        perfRecord = starter.Performance(epochs)
        
        for i in range(epochs):
            indices = np.random.permutation(n)
            featVectors = trainData[indices]
            classes = trainTarget[indices]    
            
            for j in range(0, n, batch_size):
                            
                #print(loss.eval())
                sess.run(training_op, feed_dict = {x: featVectors, y: classes})
                error = sess.run(loss, feed_dict = {x: featVectors, y: classes})
                Weights = sess.run(W, feed_dict = {x: featVectors, y: classes})
                Bias = sess.run(b, feed_dict = {x: featVectors, y: classes})                
                _, accuracy, _ = starter.classify(Weights, Bias, trainData, trainTarget)
                
        
            print('epoch #%i' % i)
            #In the process of implementing saving the performance
            #Save the per iteration errors: Error, Training Set performance, 
            #Validation Set Performance, Test Set performance 
            perfRecord.errorTrain[i] = sess.run(loss, feed_dict = {x: trainData, y: trainTarget})
            perfRecord.errorValid[i] = sess.run(loss, feed_dict = {x: validData, y: validTarget})
            perfRecord.errorTest[i] = sess.run(loss, feed_dict = {x: testData, y: testTarget})
            _, perfRecord.trainSetAcc[i] ,_ = starter.classify(Weights, Bias, trainData, trainTarget)
            _, perfRecord.validSetAcc[i] ,_ = starter.classify(Weights, Bias, validData, validTarget)
            _, perfRecord.testSetAcc[i] ,_ = starter.classify(Weights, Bias, testData, testTarget)
            
        perfRecordAll["{0}".format(epsParams[k])] = perfRecord #save the data     

#%% Plots
plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('TrainingSet Accuracy of Different Epsilon vs. Iterations')
plt.plot(perfRecordAll["1e-09"].trainSetAcc, 'r')
plt.plot(perfRecordAll["0.0001"].trainSetAcc, 'b') 
plt.legend(['Epsilon 1e-09, fin. acc = %f' % perfRecordAll["1e-09"].trainSetAcc[-1] \
            ,'Epsilon 0.0001 fin. acc = %f' % perfRecordAll["0.0001"].trainSetAcc[-1]], loc = 4)

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('ValidSet Accuracy of Different Epsilon vs. Iterations')
plt.plot(perfRecordAll["1e-09"].validSetAcc, 'r')
plt.plot(perfRecordAll["0.0001"].validSetAcc, 'b') 
plt.legend(['Epsilon 1e-09, fin. acc = %f' % perfRecordAll["1e-09"].validSetAcc[-1] \
            ,'Epsilon 0.0001 fin. acc = %f' % perfRecordAll["0.0001"].validSetAcc[-1]], loc = 4)

plt.savefig("Accuracy plot P3Q4c.png")
                      

#%% Q4b, MSE

epochs = 700
batch_size = 500
B2params = [0.99, 0.9999] 
B1=0.9; eps=1e-8;

perfRecordAll = {}
    
for k in range(0, len(B2params)):
    B2 = B2params[k]
    W, b, y_pred, x, y, loss, training_op, reg = starter.buildGraph(lossType = 'MSE', B1 = B1, B2 = B2, eps = eps)
    #Question 3, batch gradient descent
    init = tf.global_variables_initializer()
    with tf.Session() as sess:   
        sess.run(init) #make a new graph. 
        
        n = len(trainTarget)
        n_batches = int(n/batch_size)
        perfRecord = starter.Performance(epochs)
        
        for i in range(epochs):
            indices = np.random.permutation(n)
            featVectors = trainData[indices]
            classes = trainTarget[indices]    
            
            for j in range(0, n, batch_size):
                            
                #print(loss.eval())
                sess.run(training_op, feed_dict = {x: featVectors, y: classes})
                error = sess.run(loss, feed_dict = {x: featVectors, y: classes})
                Weights = sess.run(W, feed_dict = {x: featVectors, y: classes})
                Bias = sess.run(b, feed_dict = {x: featVectors, y: classes})                
                _, accuracy, _ = starter.classify(Weights, Bias, trainData, trainTarget)
                
        
            print('epoch #%i' % i)
            #In the process of implementing saving the performance
            #Save the per iteration errors: Error, Training Set performance, 
            #Validation Set Performance, Test Set performance 
            perfRecord.errorTrain[i] = sess.run(loss, feed_dict = {x: trainData, y: trainTarget})
            perfRecord.errorValid[i] = sess.run(loss, feed_dict = {x: validData, y: validTarget})
            perfRecord.errorTest[i] = sess.run(loss, feed_dict = {x: testData, y: testTarget})
            _, perfRecord.trainSetAcc[i] ,_ = starter.classify(Weights, Bias, trainData, trainTarget)
            _, perfRecord.validSetAcc[i] ,_ = starter.classify(Weights, Bias, validData, validTarget)
            _, perfRecord.testSetAcc[i] ,_ = starter.classify(Weights, Bias, testData, testTarget)
            
        perfRecordAll["{0}".format(B2params[k])] = perfRecord #save the data     

#%% Accuracy plots 

plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('TrainingSet Accuracy of Different Beta2 vs. Iterations')
plt.plot(perfRecordAll["0.99"].trainSetAcc, 'r')
plt.plot(perfRecordAll["0.9999"].trainSetAcc, 'b') 
plt.legend(['Beta2 0.99, fin. acc = %f' % perfRecordAll["0.99"].trainSetAcc[-1] \
            ,'Beta2 0.9999 fin. acc = %f' % perfRecordAll["0.9999"].trainSetAcc[-1]], loc = 4)

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('ValidSet Accuracy of Different Beta2 vs. Iterations')
plt.plot(perfRecordAll["0.99"].validSetAcc, 'r')
plt.plot(perfRecordAll["0.9999"].validSetAcc, 'b') 
plt.legend(['Beta2 0.99, fin. acc = %f' % perfRecordAll["0.99"].validSetAcc[-1] \
            ,'Beta2 0.9999 fin. acc = %f' % perfRecordAll["0.9999"].validSetAcc[-1]], loc = 4)

plt.savefig("Accuracy plot P3Q4b.png")
                      
                      
#%% Q4a, MSE

epochs = 700
batch_size = 500
B1params = [0.95, 0.99] #[500] #[100, 700, 1750]
B2=0.999; eps=1e-8;

perfRecordAll = {}
    
for k in range(0, len(B1params)):
    B1 = B1params[k]
    W, b, y_pred, x, y, loss, training_op, reg = starter.buildGraph(lossType = 'MSE', B1 = B1, B2 = B2, eps = eps)
    #Question 3, batch gradient descent
    init = tf.global_variables_initializer()
    with tf.Session() as sess:   
        sess.run(init) #make a new graph. 
        
        n = len(trainTarget)
        n_batches = int(n/batch_size)
        perfRecord = starter.Performance(epochs)
        
        for i in range(epochs):
            indices = np.random.permutation(n)
            featVectors = trainData[indices]
            classes = trainTarget[indices]    
            
            for j in range(0, n, batch_size):
                            
                #print(loss.eval())
                sess.run(training_op, feed_dict = {x: featVectors, y: classes})
                error = sess.run(loss, feed_dict = {x: featVectors, y: classes})
                Weights = sess.run(W, feed_dict = {x: featVectors, y: classes})
                Bias = sess.run(b, feed_dict = {x: featVectors, y: classes})                
                _, accuracy, _ = starter.classify(Weights, Bias, trainData, trainTarget)
                
        
            print('epoch #%i' % i)
            #In the process of implementing saving the performance
            #Save the per iteration errors: Error, Training Set performance, 
            #Validation Set Performance, Test Set performance 
            perfRecord.errorTrain[i] = sess.run(loss, feed_dict = {x: trainData, y: trainTarget})
            perfRecord.errorValid[i] = sess.run(loss, feed_dict = {x: validData, y: validTarget})
            perfRecord.errorTest[i] = sess.run(loss, feed_dict = {x: testData, y: testTarget})
            _, perfRecord.trainSetAcc[i] ,_ = starter.classify(Weights, Bias, trainData, trainTarget)
            _, perfRecord.validSetAcc[i] ,_ = starter.classify(Weights, Bias, validData, validTarget)
            _, perfRecord.testSetAcc[i] ,_ = starter.classify(Weights, Bias, testData, testTarget)
            
        perfRecordAll["{0}".format(B1params[k])] = perfRecord #save the data     


#%% Plots 

#accuracies
plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('TrainingSet Accuracy of Different Beta1 vs. Iterations')
plt.plot(perfRecordAll["0.95"].trainSetAcc, 'r')
plt.plot(perfRecordAll["0.99"].trainSetAcc, 'b') 
plt.legend(['Beta1 0.95, fin. acc = %f' % perfRecordAll["0.95"].trainSetAcc[-1] \
            ,'Beta1 0.99 fin. acc = %f' % perfRecordAll["0.99"].trainSetAcc[-1]], loc = 4)

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('ValidSet Accuracy of Different Beta1 vs. Iterations')
plt.plot(perfRecordAll["0.95"].validSetAcc, 'r')
plt.plot(perfRecordAll["0.99"].validSetAcc, 'b') 
plt.legend(['Beta1 0.95, fin. acc = %f' % perfRecordAll["0.95"].validSetAcc[-1] \
            ,'Beta1 0.99 fin. acc = %f' % perfRecordAll["0.99"].validSetAcc[-1]], loc = 4)

plt.savefig("Accuracy plot P3Q4a.png")








#%% Tensorflow part, Q3, MSE

epochs = 700
batchSizeParams = [100, 700, 1750] #[500] #[100, 700, 1750]

perfRecordAll = {}

W, b, y_pred, x, y, loss, training_op, reg = starter.buildGraph(lossType = 'MSE', B1 = 0.9, B2 = 0.999, eps = 1e-08)

#Question 3, batch gradient descent

init = tf.global_variables_initializer()
with tf.Session() as sess:   
    for k in range(0, len(batchSizeParams)):
        sess.run(init) #make a new graph. 
        
        batch_size = batchSizeParams[k]
        n = len(trainTarget)
        n_batches = int(n/batch_size)
        perfRecord = starter.Performance(epochs)
        
        for i in range(epochs):
            indices = np.random.permutation(n)
            featVectors = trainData[indices]
            classes = trainTarget[indices]    
            
            for j in range(0, n, batch_size):
                            
                #print(loss.eval())
                sess.run(training_op, feed_dict = {x: featVectors, y: classes})
                error = sess.run(loss, feed_dict = {x: featVectors, y: classes})
                Weights = sess.run(W, feed_dict = {x: featVectors, y: classes})
                Bias = sess.run(b, feed_dict = {x: featVectors, y: classes})                
                _, accuracy, _ = starter.classify(Weights, Bias, trainData, trainTarget)
                
        
            print('epoch #%i' % i)
            #In the process of implementing saving the performance
            #Save the per iteration errors: Error, Training Set performance, 
            #Validation Set Performance, Test Set performance 
            perfRecord.errorTrain[i] = sess.run(loss, feed_dict = {x: trainData, y: trainTarget})
            perfRecord.errorValid[i] = sess.run(loss, feed_dict = {x: validData, y: validTarget})
            perfRecord.errorTest[i] = sess.run(loss, feed_dict = {x: testData, y: testTarget})
            _, perfRecord.trainSetAcc[i] ,_ = starter.classify(Weights, Bias, trainData, trainTarget)
            _, perfRecord.validSetAcc[i] ,_ = starter.classify(Weights, Bias, validData, validTarget)
            _, perfRecord.testSetAcc[i] ,_ = starter.classify(Weights, Bias, testData, testTarget)
            
        perfRecordAll["{0}".format(batchSizeParams[k])] = perfRecord #save the data     
                
#%% Question 3 Tensorflow Plot #Finish this 


# Make the plots of errors and accuracies 
 #errors
plt.figure(figsize=(15,15))
plt.subplot(2, 2, 1)
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.title('TrainingSet Error of Different Batch Size vs. Iterations of Gradient Descent')
plt.plot(perfRecordAll["100"].errorTrain, 'r')
plt.plot(perfRecordAll["700"].errorTrain, 'b') 
plt.plot(perfRecordAll["1750"].errorTrain, 'g') 
plt.legend(['Batch Size 100, fin. error = %f' % perfRecordAll["100"].errorTrain[-1] \
            ,'Batch Size 700 fin. error = %f' % perfRecordAll["700"].errorTrain[-1]\
            , 'Batch Size 1750 fin. error = %f' % perfRecordAll["1750"].errorTrain[-1]])

plt.subplot(2, 2, 2)
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.title('ValidSet Error of Different Batch Size vs. Iterations of Gradient Descent')
plt.plot(perfRecordAll["100"].errorValid, 'r')
plt.plot(perfRecordAll["700"].errorValid, 'b') 
plt.plot(perfRecordAll["1750"].errorValid, 'g') 
plt.legend(['Batch Size 100, fin. error = %f' % perfRecordAll["100"].errorValid[-1] \
            ,'Batch Size 700 fin. error = %f' % perfRecordAll["700"].errorValid[-1]\
            , 'Batch Size 1750 fin. error = %f' % perfRecordAll["1750"].errorValid[-1]])

plt.subplot(2, 2, 3)
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.title('TestSet Error of Different Batch Size vs. Iterations of Gradient Descent')
plt.plot(perfRecordAll["100"].errorTest, 'r')
plt.plot(perfRecordAll["700"].errorTest, 'b') 
plt.plot(perfRecordAll["1750"].errorTest, 'g') 
plt.legend(['Batch Size 100, fin. error = %f' % perfRecordAll["100"].errorTest[-1] \
            ,'Batch Size 700 fin. error = %f' % perfRecordAll["700"].errorTest[-1]\
             , 'Batch Size 1750 fin. error = %f' % perfRecordAll["1750"].errorTest[-1]])

plt.savefig("Error plot P3Q3.png")

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

plt.savefig("Accuracy plot P3Q3.png")





#%%
#==============================================================================
#Old test code
# with tf.Session() as sess: #Used for viewing data
#       print(sess.run(data_y))
# 
#       init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     
#     for epoch in range(epochs):
#         #print(loss.eval())
#         sess.run(training_op, feed_dict = {x: trainData, y: trainTarget})
#         val = sess.run(loss, feed_dict = {x: trainData, y: trainTarget})
#         Weights = sess.run(W, feed_dict = {x: trainData, y: trainTarget})
#         Bias = sess.run(b, feed_dict = {x: trainData, y: trainTarget})
#         print('epoch #%i' % epoch)
#         _, accuracy, _ = starter.classify(Weights, Bias, trainData, trainTarget)
#         
# #set up iterator
#==============================================================================
# dataset = tf.data.Dataset.from_tensor_slices((trainData, trainTarget))
# dataset = dataset.repeat(epochs).batch(batch_size)
# iterator = dataset.make_one_shot_iterator()
# x, y = iterator.get_next()
#=====================================
#==============================================================================

#==============================================================================

 #starter.classify(Weights, Bias, testData, testTarget)


#==============================================================================
# #%% Question 3, Tuning Learning Rate
# 
# reg = 0;
# alphaParams = [0.005, 0.001, 0.0001]
# perfRecordAll = {}
# 
# for i in range(0, len(alphaParams)): 
#     #reset the model
#     W = np.random.normal(0, 0.5, size = [784,1])
#     b = 0 #np.zeros((3500,1)) #bias matrix
#     
#     alpha = alphaParams[i]
#     W,b, perfRecord = starter.grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, \
#                                            validData, validTarget, testData, testTarget )
#     
#     perfRecordAll["{0}".format(alphaParams[i])] = perfRecord #save the data
#     
#     
# #%% Make the plots of errors and accuracies 
# #errors
# plt.figure(figsize=(15,15))
# plt.subplot(2, 2, 1)
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# plt.title('TrainingSet Error of Different Alpha vs. Iterations of Gradient Descent')
# plt.plot(perfRecordAll["0.005"].errorTrain, 'r')
# plt.plot(perfRecordAll["0.001"].errorTrain, 'b') 
# plt.plot(perfRecordAll["0.0001"].errorTrain, 'g') 
# plt.legend(['alpha 0.005, fin. error = %f' % perfRecordAll["0.005"].errorTrain[-1] \
#             ,'alpha 0.001 fin. error = %f' % perfRecordAll["0.001"].errorTrain[-1]\
#             , 'alpha 0.0001 fin. error = %f' % perfRecordAll["0.0001"].errorTrain[-1]])
# 
# plt.subplot(2, 2, 2)
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# plt.title('ValidSet Error of Different Alpha vs. Iterations of Gradient Descent')
# plt.plot(perfRecordAll["0.005"].errorValid, 'r')
# plt.plot(perfRecordAll["0.001"].errorValid, 'b') 
# plt.plot(perfRecordAll["0.0001"].errorValid, 'g') 
# plt.legend(['alpha 0.005, fin. error = %f' % perfRecordAll["0.005"].errorValid[-1] \
#             ,'alpha 0.001 fin. error = %f' % perfRecordAll["0.001"].errorValid[-1]\
#             , 'alpha 0.0001 fin. error = %f' % perfRecordAll["0.0001"].errorValid[-1]])
# 
# plt.subplot(2, 2, 3)
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# plt.title('TestSet Error of Different Alpha vs. Iterations of Gradient Descent')
# plt.plot(perfRecordAll["0.005"].errorTest, 'r')
# plt.plot(perfRecordAll["0.001"].errorTest, 'b') 
# plt.plot(perfRecordAll["0.0001"].errorTest, 'g') 
# plt.legend(['alpha 0.005, fin. error = %f' % perfRecordAll["0.005"].errorTest[-1] \
#             ,'alpha 0.001 fin. error = %f' % perfRecordAll["0.001"].errorTest[-1]\
#             , 'alpha 0.0001 fin. error = %f' % perfRecordAll["0.0001"].errorTest[-1]])
# 
# 
# #accuracies
# plt.figure(figsize=(15,15))
# plt.subplot(2, 2, 1)
# plt.ylabel('Accuracy')
# plt.xlabel('Iterations')
# plt.title('TrainingSet Accuracy of Different Alpha vs. Iterations')
# plt.plot(perfRecordAll["0.005"].trainSetAcc, 'r')
# plt.plot(perfRecordAll["0.001"].trainSetAcc, 'b') 
# plt.plot(perfRecordAll["0.0001"].trainSetAcc, 'g') 
# plt.legend(['alpha 0.005, fin. acc = %f' % perfRecordAll["0.005"].trainSetAcc[-1] \
#             ,'alpha 0.001 fin. acc = %f' % perfRecordAll["0.001"].trainSetAcc[-1]\
#             , 'alpha 0.0001 fin. acc = %f' % perfRecordAll["0.0001"].trainSetAcc[-1]], loc = 4)
# 
# plt.subplot(2, 2, 2)
# plt.ylabel('Accuracy')
# plt.xlabel('Iterations')
# plt.title('ValidSet Accuracy of Different Alpha vs. Iterations')
# plt.plot(perfRecordAll["0.005"].validSetAcc, 'r')
# plt.plot(perfRecordAll["0.001"].validSetAcc, 'b') 
# plt.plot(perfRecordAll["0.0001"].validSetAcc, 'g') 
# plt.legend(['alpha 0.005, fin. acc = %f' % perfRecordAll["0.005"].validSetAcc[-1] \
#             ,'alpha 0.001 fin. acc = %f' % perfRecordAll["0.001"].validSetAcc[-1]\
#             , 'alpha 0.0001 fin. acc = %f' % perfRecordAll["0.0001"].validSetAcc[-1]], loc = 4)
# 
# plt.subplot(2, 2, 3)
# plt.ylabel('Accuracy')
# plt.xlabel('Iterations')
# plt.title('TestSet Accuracy of Different Alpha vs. Iterations o')
# plt.plot(perfRecordAll["0.005"].testSetAcc, 'r')
# plt.plot(perfRecordAll["0.001"].testSetAcc, 'b') 
# plt.plot(perfRecordAll["0.0001"].testSetAcc, 'g') 
# plt.legend(['alpha 0.005, fin. acc = %f' % perfRecordAll["0.005"].testSetAcc[-1] \
#             ,'alpha 0.001 fin. acc = %f' % perfRecordAll["0.001"].testSetAcc[-1]\
#             , 'alpha 0.0001 fin. acc = %f' % perfRecordAll["0.0001"].testSetAcc[-1]], loc = 4)
# 
# 
# 
# 
# #%% Question 4, 
# 
# alpha = 0.005;
# regParams = [0.001, 0.1, 0.5]
# perfRecordAll = {}
# 
# for i in range(0, len(regParams)): 
#     #reset the model
# 
#     W = np.random.normal(0, 0.5, size = [784,1])
#     b = 0#np.zeros((3500,1)) #bias matrix
#     
#     reg = regParams[i]
#     W,b, perfRecord = starter.grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, \
#                                            validData, validTarget, testData, testTarget )
#     
#     perfRecordAll["{0}".format(regParams[i])] = perfRecord #save the data
#     
#     
# #%% Make the plots of errors and accuracies 
# #errors
# plt.figure(figsize=(15,15))
# plt.subplot(2, 2, 1)
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# plt.title('TrainingSet Error of Different Reg vs. Iterations of Gradient Descent')
# plt.plot(perfRecordAll["0.001"].errorTrain, 'r')
# plt.plot(perfRecordAll["0.1"].errorTrain, 'b') 
# plt.plot(perfRecordAll["0.5"].errorTrain, 'g') 
# plt.legend(['reg 0.001, fin. error = %f' % perfRecordAll["0.001"].errorTrain[-1] \
#             ,'reg 0.1 fin. error = %f' % perfRecordAll["0.1"].errorTrain[-1]\
#             , 'reg 0.5 fin. error = %f' % perfRecordAll["0.5"].errorTrain[-1]])
# 
# plt.subplot(2, 2, 2)
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# plt.title('ValidSet Error of Different Reg vs. Iterations of Gradient Descent')
# plt.plot(perfRecordAll["0.001"].errorValid, 'r')
# plt.plot(perfRecordAll["0.1"].errorValid, 'b') 
# plt.plot(perfRecordAll["0.5"].errorValid, 'g') 
# plt.legend(['reg 0.001, fin. error = %f' % perfRecordAll["0.001"].errorValid[-1] \
#             ,'reg 0.1 fin. error = %f' % perfRecordAll["0.1"].errorValid[-1]\
#             , 'reg 0.5 fin. error = %f' % perfRecordAll["0.5"].errorValid[-1]])
# 
# plt.subplot(2, 2, 3)
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# plt.title('TestSet Error of Different reg vs. Iterations of Gradient Descent')
# plt.plot(perfRecordAll["0.001"].errorTest, 'r')
# plt.plot(perfRecordAll["0.1"].errorTest, 'b') 
# plt.plot(perfRecordAll["0.5"].errorTest, 'g') 
# plt.legend(['reg 0.001, fin. error = %f' % perfRecordAll["0.001"].errorTest[-1] \
#             ,'reg 0.1 fin. error = %f' % perfRecordAll["0.1"].errorTest[-1]\
#             , 'reg 0.5 fin. error = %f' % perfRecordAll["0.5"].errorTest[-1]])
# 
# 
# #accuracies
# plt.figure(figsize=(15,15))
# plt.subplot(2, 2, 1)
# plt.ylabel('Accuracy')
# plt.xlabel('Iterations')
# plt.title('TrainingSet Accuracy of Different reg vs. Iterations')
# plt.plot(perfRecordAll["0.001"].trainSetAcc, 'r')
# plt.plot(perfRecordAll["0.1"].trainSetAcc, 'b') 
# plt.plot(perfRecordAll["0.5"].trainSetAcc, 'g') 
# plt.legend(['reg 0.001, fin. acc = %f' % perfRecordAll["0.001"].trainSetAcc[-1] \
#             ,'reg 0.1 fin. acc = %f' % perfRecordAll["0.1"].trainSetAcc[-1]\
#             , 'reg 0.5 fin. acc = %f' % perfRecordAll["0.5"].trainSetAcc[-1]], loc = 4)
# 
# plt.subplot(2, 2, 2)
# plt.ylabel('Accuracy')
# plt.xlabel('Iterations')
# plt.title('ValidSet Accuracy of Different reg vs. Iterations')
# plt.plot(perfRecordAll["0.001"].validSetAcc, 'r')
# plt.plot(perfRecordAll["0.1"].validSetAcc, 'b') 
# plt.plot(perfRecordAll["0.5"].validSetAcc, 'g') 
# plt.legend(['reg 0.001, fin. acc = %f' % perfRecordAll["0.001"].validSetAcc[-1] \
#             ,'reg 0.1 fin. acc = %f' % perfRecordAll["0.1"].validSetAcc[-1]\
#             , 'reg 0.5 fin. acc = %f' % perfRecordAll["0.5"].validSetAcc[-1]], loc = 4)
# 
# plt.subplot(2, 2, 3)
# plt.ylabel('Accuracy')
# plt.xlabel('Iterations')
# plt.title('TestSet Accuracy of Different reg vs. Iterations')
# plt.plot(perfRecordAll["0.001"].testSetAcc, 'r')
# plt.plot(perfRecordAll["0.1"].testSetAcc, 'b') 
# plt.plot(perfRecordAll["0.5"].testSetAcc, 'g') 
# plt.legend(['reg 0.001, fin. acc = %f' % perfRecordAll["0.001"].testSetAcc[-1] \
#             ,'reg 0.1 fin. acc = %f' % perfRecordAll["0.1"].testSetAcc[-1]\
#             , 'reg 0.5 fin. acc = %f' % perfRecordAll["0.5"].testSetAcc[-1]], loc = 4)
# 
# 
# #%% Test
# plt.figure(0)
# 
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# plt.title('Reg = 0.5, Train/Valid/Test set error vs. Iterations')
# plt.plot(perfRecordAll["0.5"].errorTrain, 'r')
# plt.plot(perfRecordAll["0.5"].errorValid, 'b') 
# plt.plot(perfRecordAll["0.5"].errorTest, 'g') 
# plt.legend(['Train, fin. error = %f' % perfRecordAll["0.5"].errorTrain[-1] \
#             ,'Valid fin. error = %f' % perfRecordAll["0.5"].errorValid[-1]\
#             , 'Test fin. error = %f' % perfRecordAll["0.5"].errorTest[-1]], loc = 1)
# 
# plt.ylabel('Error')
# plt.xlabel('Iterations')
# plt.title('Reg = 0.001, Train/Valid/Test set error vs. Iterations')
# plt.plot(perfRecordAll["0.001"].errorTrain, 'r')
# plt.plot(perfRecordAll["0.001"].errorValid, 'b') 
# plt.plot(perfRecordAll["0.001"].errorTest, 'g') 
# plt.legend(['Train, fin. error = %f' % perfRecordAll["0.001"].errorTrain[-1] \
#             ,'Valid fin. error = %f' % perfRecordAll["0.001"].errorValid[-1]\
#             , 'Test fin. error = %f' % perfRecordAll["0.001"].errorTest[-1]], loc = 1)
# 
# 
# #%% Question 5 
# 
# #Calculate the optimal using normal equation
# #add a col to represent bias
# 
# tic = time.clock()
# X = np.ones([3500,1]) 
# X = np.concatenate((X, trainData), axis = 1)
# Xdagger = (np.matmul(X.T, X))
# Xdagger = np.linalg.inv(Xdagger)
# Xdagger = np.matmul(Xdagger, X.T)
# Wopt = np.matmul(Xdagger, trainTarget)
# bias = Wopt[0]
# Wopt = Wopt[1:]
# 
# 
# _, acc ,_ = starter.classify(Wopt, bias , trainData, trainTarget)
# 
# toc = time.clock() - tic
# print(toc)
#==============================================================================

#%% Old test code below
#Check classifications
#perform a classification
#positive class is C, negative is J
#==============================================================================
# letterIndex = 77;
# print('classification is %f' % (np.matmul(testData[letterIndex,:], W) + b))
# plt.imshow(origTestData[letterIndex,:,:], cmap = 'gray')
#==============================================================================

#==============================================================================
# value = starter.MSE(W, b, trainData, trainTarget, reg) #call MSE that I just implemented, but I need to check it
# grad_wrtb, grad_wrtW = starter.gradMSE(W, b, trainData, trainTarget, reg)
# W,b, perfRecord = starter.grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, \
#                            validData, validTarget, testData, testTarget )
# y_hat, accuracy, misclassIndices = starter.classify(W, b, testData, testTarget)
#==============================================================================


#very small test data
#==============================================================================
# W = np.matrix('1;1')
# b = 1
# trainData = np.matrix('1,1; 2,2; 3,3')
# trainTarget = np.matrix('1;0;1')
# reg = 0
#==============================================================================
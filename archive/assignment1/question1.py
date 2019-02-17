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


#%% Question 3, Tuning Learning Rate
def plot_1_3():
     reg = 0;
     alphaParams = [0.005, 0.001, 0.0001]
     perfRecordAll = {}
     
     for i in range(0, len(alphaParams)): 
         #reset the model
         W = np.random.normal(0, 0.5, size = [784,1])
         b = 0 #np.zeros((3500,1)) #bias matrix
         
         alpha = alphaParams[i]
         W,b, perfRecord = starter.grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, \
                                                validData, validTarget, testData, testTarget )
         
         perfRecordAll["{0}".format(alphaParams[i])] = perfRecord #save the data
         
         
     #%% Make the plots of errors and accuracies 
     #errors
     plt.figure(figsize=(15,15))
     plt.subplot(2, 2, 1)
     plt.ylabel('Error')
     plt.xlabel('Iterations')
     plt.title('TrainingSet Error of Different Alpha vs. Iterations of Gradient Descent')
     plt.plot(perfRecordAll["0.005"].errorTrain, 'r')
     plt.plot(perfRecordAll["0.001"].errorTrain, 'b') 
     plt.plot(perfRecordAll["0.0001"].errorTrain, 'g') 
     plt.legend(['alpha 0.005, fin. error = %f' % perfRecordAll["0.005"].errorTrain[-1] \
                 ,'alpha 0.001 fin. error = %f' % perfRecordAll["0.001"].errorTrain[-1]\
                 , 'alpha 0.0001 fin. error = %f' % perfRecordAll["0.0001"].errorTrain[-1]])
     
     plt.subplot(2, 2, 2)
     plt.ylabel('Error')
     plt.xlabel('Iterations')
     plt.title('ValidSet Error of Different Alpha vs. Iterations of Gradient Descent')
     plt.plot(perfRecordAll["0.005"].errorValid, 'r')
     plt.plot(perfRecordAll["0.001"].errorValid, 'b') 
     plt.plot(perfRecordAll["0.0001"].errorValid, 'g') 
     plt.legend(['alpha 0.005, fin. error = %f' % perfRecordAll["0.005"].errorValid[-1] \
                 ,'alpha 0.001 fin. error = %f' % perfRecordAll["0.001"].errorValid[-1]\
                 , 'alpha 0.0001 fin. error = %f' % perfRecordAll["0.0001"].errorValid[-1]])
     
     plt.subplot(2, 2, 3)
     plt.ylabel('Error')
     plt.xlabel('Iterations')
     plt.title('TestSet Error of Different Alpha vs. Iterations of Gradient Descent')
     plt.plot(perfRecordAll["0.005"].errorTest, 'r')
     plt.plot(perfRecordAll["0.001"].errorTest, 'b') 
     plt.plot(perfRecordAll["0.0001"].errorTest, 'g') 
     plt.legend(['alpha 0.005, fin. error = %f' % perfRecordAll["0.005"].errorTest[-1] \
                 ,'alpha 0.001 fin. error = %f' % perfRecordAll["0.001"].errorTest[-1]\
                 , 'alpha 0.0001 fin. error = %f' % perfRecordAll["0.0001"].errorTest[-1]])
     
     
     #accuracies
     plt.figure(figsize=(15,15))
     plt.subplot(2, 2, 1)
     plt.ylabel('Accuracy')
     plt.xlabel('Iterations')
     plt.title('TrainingSet Accuracy of Different Alpha vs. Iterations')
     plt.plot(perfRecordAll["0.005"].trainSetAcc, 'r')
     plt.plot(perfRecordAll["0.001"].trainSetAcc, 'b') 
     plt.plot(perfRecordAll["0.0001"].trainSetAcc, 'g') 
     plt.legend(['alpha 0.005, fin. acc = %f' % perfRecordAll["0.005"].trainSetAcc[-1] \
                 ,'alpha 0.001 fin. acc = %f' % perfRecordAll["0.001"].trainSetAcc[-1]\
                 , 'alpha 0.0001 fin. acc = %f' % perfRecordAll["0.0001"].trainSetAcc[-1]], loc = 4)
     
     plt.subplot(2, 2, 2)
     plt.ylabel('Accuracy')
     plt.xlabel('Iterations')
     plt.title('ValidSet Accuracy of Different Alpha vs. Iterations')
     plt.plot(perfRecordAll["0.005"].validSetAcc, 'r')
     plt.plot(perfRecordAll["0.001"].validSetAcc, 'b') 
     plt.plot(perfRecordAll["0.0001"].validSetAcc, 'g') 
     plt.legend(['alpha 0.005, fin. acc = %f' % perfRecordAll["0.005"].validSetAcc[-1] \
                 ,'alpha 0.001 fin. acc = %f' % perfRecordAll["0.001"].validSetAcc[-1]\
                 , 'alpha 0.0001 fin. acc = %f' % perfRecordAll["0.0001"].validSetAcc[-1]], loc = 4)
     
     plt.subplot(2, 2, 3)
     plt.ylabel('Accuracy')
     plt.xlabel('Iterations')
     plt.title('TestSet Accuracy of Different Alpha vs. Iterations o')
     plt.plot(perfRecordAll["0.005"].testSetAcc, 'r')
     plt.plot(perfRecordAll["0.001"].testSetAcc, 'b') 
     plt.plot(perfRecordAll["0.0001"].testSetAcc, 'g') 
     plt.legend(['alpha 0.005, fin. acc = %f' % perfRecordAll["0.005"].testSetAcc[-1] \
                 ,'alpha 0.001 fin. acc = %f' % perfRecordAll["0.001"].testSetAcc[-1]\
                 , 'alpha 0.0001 fin. acc = %f' % perfRecordAll["0.0001"].testSetAcc[-1]], loc = 4)
     

#%% Question 4, 
def plot_1_4():
     alpha = 0.005;
     regParams = [0.001, 0.1, 0.5]
     perfRecordAll = {}
     
     for i in range(0, len(regParams)): 
         #reset the model
     
         W = np.random.normal(0, 0.5, size = [784,1])
         b = 0#np.zeros((3500,1)) #bias matrix
         
         reg = regParams[i]
         W,b, perfRecord = starter.grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, \
                                                validData, validTarget, testData, testTarget )
         
         perfRecordAll["{0}".format(regParams[i])] = perfRecord #save the data
         
         
     #%% Make the plots of errors and accuracies 
     #errors
     plt.figure(figsize=(15,15))
     plt.subplot(2, 2, 1)
     plt.ylabel('Error')
     plt.xlabel('Iterations')
     plt.title('TrainingSet Error of Different Reg vs. Iterations of Gradient Descent')
     plt.plot(perfRecordAll["0.001"].errorTrain, 'r')
     plt.plot(perfRecordAll["0.1"].errorTrain, 'b') 
     plt.plot(perfRecordAll["0.5"].errorTrain, 'g') 
     plt.legend(['reg 0.001, fin. error = %f' % perfRecordAll["0.001"].errorTrain[-1] \
                 ,'reg 0.1 fin. error = %f' % perfRecordAll["0.1"].errorTrain[-1]\
                 , 'reg 0.5 fin. error = %f' % perfRecordAll["0.5"].errorTrain[-1]])
     
     plt.subplot(2, 2, 2)
     plt.ylabel('Error')
     plt.xlabel('Iterations')
     plt.title('ValidSet Error of Different Reg vs. Iterations of Gradient Descent')
     plt.plot(perfRecordAll["0.001"].errorValid, 'r')
     plt.plot(perfRecordAll["0.1"].errorValid, 'b') 
     plt.plot(perfRecordAll["0.5"].errorValid, 'g') 
     plt.legend(['reg 0.001, fin. error = %f' % perfRecordAll["0.001"].errorValid[-1] \
                 ,'reg 0.1 fin. error = %f' % perfRecordAll["0.1"].errorValid[-1]\
                 , 'reg 0.5 fin. error = %f' % perfRecordAll["0.5"].errorValid[-1]])
     
     plt.subplot(2, 2, 3)
     plt.ylabel('Error')
     plt.xlabel('Iterations')
     plt.title('TestSet Error of Different reg vs. Iterations of Gradient Descent')
     plt.plot(perfRecordAll["0.001"].errorTest, 'r')
     plt.plot(perfRecordAll["0.1"].errorTest, 'b') 
     plt.plot(perfRecordAll["0.5"].errorTest, 'g') 
     plt.legend(['reg 0.001, fin. error = %f' % perfRecordAll["0.001"].errorTest[-1] \
                 ,'reg 0.1 fin. error = %f' % perfRecordAll["0.1"].errorTest[-1]\
                 , 'reg 0.5 fin. error = %f' % perfRecordAll["0.5"].errorTest[-1]])
     
     
     #accuracies
     plt.figure(figsize=(15,15))
     plt.subplot(2, 2, 1)
     plt.ylabel('Accuracy')
     plt.xlabel('Iterations')
     plt.title('TrainingSet Accuracy of Different reg vs. Iterations')
     plt.plot(perfRecordAll["0.001"].trainSetAcc, 'r')
     plt.plot(perfRecordAll["0.1"].trainSetAcc, 'b') 
     plt.plot(perfRecordAll["0.5"].trainSetAcc, 'g') 
     plt.legend(['reg 0.001, fin. acc = %f' % perfRecordAll["0.001"].trainSetAcc[-1] \
                 ,'reg 0.1 fin. acc = %f' % perfRecordAll["0.1"].trainSetAcc[-1]\
                 , 'reg 0.5 fin. acc = %f' % perfRecordAll["0.5"].trainSetAcc[-1]], loc = 4)
     
     plt.subplot(2, 2, 2)
     plt.ylabel('Accuracy')
     plt.xlabel('Iterations')
     plt.title('ValidSet Accuracy of Different reg vs. Iterations')
     plt.plot(perfRecordAll["0.001"].validSetAcc, 'r')
     plt.plot(perfRecordAll["0.1"].validSetAcc, 'b') 
     plt.plot(perfRecordAll["0.5"].validSetAcc, 'g') 
     plt.legend(['reg 0.001, fin. acc = %f' % perfRecordAll["0.001"].validSetAcc[-1] \
                 ,'reg 0.1 fin. acc = %f' % perfRecordAll["0.1"].validSetAcc[-1]\
                 , 'reg 0.5 fin. acc = %f' % perfRecordAll["0.5"].validSetAcc[-1]], loc = 4)
     
     plt.subplot(2, 2, 3)
     plt.ylabel('Accuracy')
     plt.xlabel('Iterations')
     plt.title('TestSet Accuracy of Different reg vs. Iterations')
     plt.plot(perfRecordAll["0.001"].testSetAcc, 'r')
     plt.plot(perfRecordAll["0.1"].testSetAcc, 'b') 
     plt.plot(perfRecordAll["0.5"].testSetAcc, 'g') 
     plt.legend(['reg 0.001, fin. acc = %f' % perfRecordAll["0.001"].testSetAcc[-1] \
                 ,'reg 0.1 fin. acc = %f' % perfRecordAll["0.1"].testSetAcc[-1]\
                 , 'reg 0.5 fin. acc = %f' % perfRecordAll["0.5"].testSetAcc[-1]], loc = 4)
     
     #%% Test
     plt.figure(0)
     
     plt.ylabel('Error')
     plt.xlabel('Iterations')
     plt.title('Reg = 0.5, Train/Valid/Test set error vs. Iterations')
     plt.plot(perfRecordAll["0.5"].errorTrain, 'r')
     plt.plot(perfRecordAll["0.5"].errorValid, 'b') 
     plt.plot(perfRecordAll["0.5"].errorTest, 'g') 
     plt.legend(['Train, fin. error = %f' % perfRecordAll["0.5"].errorTrain[-1] \
                 ,'Valid fin. error = %f' % perfRecordAll["0.5"].errorValid[-1]\
                 , 'Test fin. error = %f' % perfRecordAll["0.5"].errorTest[-1]], loc = 1)
     
     plt.ylabel('Error')
     plt.xlabel('Iterations')
     plt.title('Reg = 0.001, Train/Valid/Test set error vs. Iterations')
     plt.plot(perfRecordAll["0.001"].errorTrain, 'r')
     plt.plot(perfRecordAll["0.001"].errorValid, 'b') 
     plt.plot(perfRecordAll["0.001"].errorTest, 'g') 
     plt.legend(['Train, fin. error = %f' % perfRecordAll["0.001"].errorTrain[-1] \
                 ,'Valid fin. error = %f' % perfRecordAll["0.001"].errorValid[-1]\
                 , 'Test fin. error = %f' % perfRecordAll["0.001"].errorTest[-1]], loc = 1)
     
    
#%% Question 5 
def plot_1_5():
     
     #Calculate the optimal using normal equation
     #add a col to represent bias
     
     tic = time.clock()
     X = np.ones([3500,1]) 
     X = np.concatenate((X, trainData), axis = 1)
     Xdagger = (np.matmul(X.T, X))
     Xdagger = np.linalg.inv(Xdagger)
     Xdagger = np.matmul(Xdagger, X.T)
     Wopt = np.matmul(Xdagger, trainTarget)
     bias = Wopt[0]
     Wopt = Wopt[1:]
     
     
     _, acc ,_ = starter.classify(Wopt, bias , trainData, trainTarget)
     
     toc = time.clock() - tic
     print(toc)
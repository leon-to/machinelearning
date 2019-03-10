# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:09:45 2019

@author: Michael
"""

import starter as starter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#%% Load Data and One-hot ecoding and parameters
trainData, validData, testData, trainTarget, validTarget, testTarget = starter.loadData()
#Reshape data to vectors
trainData = trainData.reshape([trainData.shape[0],784]) #train data matrix
testData = testData.reshape([testData.shape[0],784]) #test data matrix
validData = validData.reshape([validData.shape[0],784]) #validation data matrix

# Change from labels to one-hot encoding
trainTarget, validTarget, testTarget = starter.convertOneHot(trainTarget, validTarget, testTarget)

K = 500 #number of hidden nodes
epochs = 200


#%% 1.3 Create NN

#Xavier weight initialization parameters
numClasses = testTarget.shape[1] #10
numFeature = trainData.shape[1] #1000

Wh = np.random.normal(0, (2/(numFeature + K)), (numFeature, K))
bh = np.random.normal(0, 0.01, (1, K)) #normal dist with small variance
Wo = np.random.normal(0, (2/(testTarget.shape[1] + K)), (K, numClasses)) #(testTarget.shape[1] + K) is number of classes
bo = np.random.normal(0, 0.01, (1, (numClasses)))
Vh =  np.full((numFeature, K), 1e-5)
Vo =  np.full((K, numClasses), 1e-5)

alpha = 0.01#0.1 #Leaning Rate
gamma = 0.9
Vold_h = 0
Vold_o = 0
Vnew_h = 0
Vnew_o = 0

costs = []
accuracy = []

for epoch in range(epochs):
    #Forward Pass
    Z1 = np.matmul(trainData, Wh)  #sums of hidden layer, in rows
    X1 = starter.relu(Z1 + bh) #outputs of hidden layyer
    Z2 = np.matmul(X1, Wo) #sums of output layer, in rows
    Y = starter.softmax((Z2 + bo).T).T
    
    #Backward pass
    delta2 = Y - trainTarget
    delta1 = (np.matmul(delta2,Wo.T))*np.heaviside(Z1,1)
    
    #Weight update
    Vnew_o = gamma*Vold_o + alpha*X1.T.dot(delta2)
    Vold_o = Vnew_o
    bo =  bo - alpha*(delta2).sum(axis = 0)
    Wh = Wh - Vnew_h
    
    Vnew_h = gamma*Vold_h + alpha*trainData.T.dot(delta1)
    Vold_h = Vnew_h
    Wh = Wh - Vnew_h
    bh = bh - alpha*(delta1).sum(axis=0)
    
    loss = starter.CE(trainTarget, Y)
    costs.append(loss)
    
    #Test score
    predict = starter.forwardPass(testData, Wh, Wo, bh, bo);
    targets = np.argmax(testTarget, axis = 1)    
    accuracy = np.mean(predict == targets)
    testAcc = accuracy
    
    #Train score
    predict = starter.forwardPass(trainData, Wh, Wo, bh, bo);
    targets = np.argmax(trainTarget, axis = 1)    
    accuracy = np.mean(predict == targets)
    
    trainAcc = accuracy

    print('Epoch is', epoch,'Loss is ', loss, 'Test Acc is ', testAcc, 'Train Acc is', trainAcc)
    
    
    
    
#%%
plt.plot(costs)


#%%  Tests
starter.relu(validTarget)

predictions = np.array([[0.25,0.25,0.25,0.25],
                        [0.01,0.01,0.01,0.97]])
targets = np.array([[1,0,0,0],
                   [0,0,0,1]])
d = starter.CE(targets,predictions)
e = starter.gradCE(targets,predictions)
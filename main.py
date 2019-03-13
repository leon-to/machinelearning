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



#%% 1.3 Create NN

#Xavier weight initialization parameters
K = 1000 #number of hidden nodes
epochs = 200

numClasses = testTarget.shape[1] #10
numFeature = trainData.shape[1] #1000

Wh = np.random.normal(0, (2/(numFeature + K)), (numFeature, K))
bh = np.random.normal(0, 0.01, (1, K)) #normal dist with small variance
Wo = np.random.normal(0, (2/(testTarget.shape[1] + K)), (K, numClasses)) #(testTarget.shape[1] + K) is number of classes
bo = np.random.normal(0, 0.01, (1, (numClasses)))
Vh =  np.full((numFeature, K), 1e-5)
Vo =  np.full((K, numClasses), 1e-5)

alpha = 0.01#0.1 #Leaning Rate
gamma = 0.99
Vold_h = 0
Vold_o = 0
Vnew_h = 0
Vnew_o = 0

costs = []
accuracy = []

perfRecord = starter.Performance()

for epoch in range(epochs):
    #Forward Pass
    Z1 = np.matmul(trainData, Wh)  #sums of hidden layer, in rows
    X1 = starter.relu(Z1 + bh) #outputs of hidden layyer
    Z2 = np.matmul(X1, Wo) #sums of output layer, in rows
    Y = starter.softmax((Z2 + bo).T).T #final outputs
    
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
    YoneHot, predict = starter.forwardPass(testData, Wh, Wo, bh, bo);
    targets = np.argmax(testTarget, axis = 1)    
    testAccuracy = np.mean(predict == targets)
    testLoss = starter.CE(testTarget, YoneHot)
    
    #Train score
    YoneHot, predict = starter.forwardPass(trainData, Wh, Wo, bh, bo);
    targets = np.argmax(trainTarget, axis = 1)    
    trainLoss = starter.CE(trainTarget, YoneHot)
    trainAccuracy = np.mean(predict == targets)
    
    #Valid Score
    YoneHot, predict = starter.forwardPass(validData, Wh, Wo, bh, bo);
    targets = np.argmax(validTarget, axis = 1)    
    validAccuracy = np.mean(predict == targets)
    validLoss = starter.CE(validTarget, YoneHot)
    
    perfRecord.errorTrain.append(trainLoss)
    perfRecord.errorValid.append(validLoss)
    perfRecord.errorTest.append(testLoss)
    
    perfRecord.trainSetAcc.append(trainAccuracy)
    perfRecord.validSetAcc.append(validAccuracy)
    perfRecord.testSetAcc.append(testAccuracy)
    
    print('Epoch is', epoch,'Loss is ', loss, 'Test Acc is ', testAccuracy, 'Train Acc is', trainAccuracy)
    
    
    
    
#%% Part 1 plots

#Accuracy plots
plt.figure(figsize=(15,15))
plt.subplot(2, 1, 1)
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('Training, Valid and Test Set Accuracy of K = 1000')
plt.plot(perfRecord.trainSetAcc, 'r')
plt.plot(perfRecord.validSetAcc, 'g')
plt.plot(perfRecord.testSetAcc, 'b')
plt.legend(['Training Set, fin acc = %f' % perfRecord.trainSetAcc[-1] 
            , 'Valid Set, fin acc = %f' % perfRecord.validSetAcc[-1] 
            ,'Test Set, fin acc = %f' % perfRecord.testSetAcc[-1], ], loc = 4)

#Loss plots
plt.subplot(2, 1, 2)
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.title('Training, Valid and Test Set Loss of K = 1000')
plt.plot(perfRecord.errorTrain, 'r')
plt.plot(perfRecord.errorValid, 'g')
plt.plot(perfRecord.errorTest, 'b')
plt.legend(['Training Set, fin loss = %f' % perfRecord.errorTrain[-1] 
            , 'Valid Set, fin loss = %f' % perfRecord.errorValid[-1] 
            ,'Test Set, fin loss = %f' % perfRecord.errorTest[-1], ], loc = 1)

plt.savefig("Accuracy and loss plots 1.3.png")
#%% 1.4 Hypterparameter invest

params = [100, 500, 2000]

for K in params:
    
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
    
    perfRecord = starter.Performance()
    
    for epoch in range(epochs):
        #Forward Pass
        Z1 = np.matmul(trainData, Wh)  #sums of hidden layer, in rows
        X1 = starter.relu(Z1 + bh) #outputs of hidden layyer
        Z2 = np.matmul(X1, Wo) #sums of output layer, in rows
        Y = starter.softmax((Z2 + bo).T).T #final outputs
        
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
        YoneHot, predict = starter.forwardPass(testData, Wh, Wo, bh, bo);
        targets = np.argmax(testTarget, axis = 1)    
        testAccuracy = np.mean(predict == targets)
        testLoss = starter.CE(testTarget, YoneHot)
        
        #Train score
        YoneHot, predict = starter.forwardPass(trainData, Wh, Wo, bh, bo);
        targets = np.argmax(trainTarget, axis = 1)    
        trainLoss = starter.CE(trainTarget, YoneHot)
        trainAccuracy = np.mean(predict == targets)
        
        #Valid Score
        YoneHot, predict = starter.forwardPass(validData, Wh, Wo, bh, bo);
        targets = np.argmax(validTarget, axis = 1)    
        validAccuracy = np.mean(predict == targets)
        validLoss = starter.CE(validTarget, YoneHot)
        
        perfRecord.errorTrain.append(trainLoss)
        perfRecord.errorValid.append(validLoss)
        perfRecord.errorTest.append(testLoss)
        
        perfRecord.trainSetAcc.append(trainAccuracy)
        perfRecord.validSetAcc.append(validAccuracy)
        perfRecord.testSetAcc.append(testAccuracy)
        
        print('Epoch is', epoch,'Loss is ', loss, 'Test Acc is ', testAccuracy, 'Train Acc is', trainAccuracy)
        
    if K == 100:
        perfRecord100 = perfRecord
    elif K == 500:
        perfRecord500 = perfRecord
    elif K == 2000:
        perfRecord2000 = perfRecord
    
#%% 
        
#Accuracy plots
plt.figure(figsize=(15,15))
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('Training, Valid and Test Set Accuracy of different units')
plt.plot(perfRecord100.testSetAcc, 'r')
plt.plot(perfRecord500.testSetAcc, 'g')
plt.plot(perfRecord2000.testSetAcc, 'b')
plt.legend(['100 units, fin acc = %f' % perfRecord100.testSetAcc[-1] 
            , '500 units, fin acc = %f' % perfRecord500.testSetAcc[-1] 
            ,'2000 units, fin acc = %f' % perfRecord2000.testSetAcc[-1], ], loc = 4)
        
plt.savefig("Accuracy 1.4a.png")   





#%% 1.4b Early stopping

#Xavier weight initialization parameters
K = 1000 #number of hidden nodes


numClasses = testTarget.shape[1] #10
numFeature = trainData.shape[1] #1000

Wh = np.random.normal(0, (2/(numFeature + K)), (numFeature, K))
bh = np.random.normal(0, 0.01, (1, K)) #normal dist with small variance
Wo = np.random.normal(0, (2/(testTarget.shape[1] + K)), (K, numClasses)) #(testTarget.shape[1] + K) is number of classes
bo = np.random.normal(0, 0.01, (1, (numClasses)))
Vh =  np.full((numFeature, K), 1e-5)
Vo =  np.full((K, numClasses), 1e-5)

alpha = 0.01#0.1 #Leaning Rate
gamma = 0.99
Vold_h = 0
Vold_o = 0
Vnew_h = 0
Vnew_o = 0

costs = []
accuracy = []

perfRecord = starter.Performance()

epochs = 80
for epoch in range(epochs):
    #Forward Pass
    Z1 = np.matmul(trainData, Wh)  #sums of hidden layer, in rows
    X1 = starter.relu(Z1 + bh) #outputs of hidden layyer
    Z2 = np.matmul(X1, Wo) #sums of output layer, in rows
    Y = starter.softmax((Z2 + bo).T).T #final outputs
    
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
    YoneHot, predict = starter.forwardPass(testData, Wh, Wo, bh, bo);
    targets = np.argmax(testTarget, axis = 1)    
    testAccuracy = np.mean(predict == targets)
    testLoss = starter.CE(testTarget, YoneHot)
    
    #Train score
    YoneHot, predict = starter.forwardPass(trainData, Wh, Wo, bh, bo);
    targets = np.argmax(trainTarget, axis = 1)    
    trainLoss = starter.CE(trainTarget, YoneHot)
    trainAccuracy = np.mean(predict == targets)
    
    #Valid Score
    YoneHot, predict = starter.forwardPass(validData, Wh, Wo, bh, bo);
    targets = np.argmax(validTarget, axis = 1)    
    validAccuracy = np.mean(predict == targets)
    validLoss = starter.CE(validTarget, YoneHot)
    
    perfRecord.errorTrain.append(trainLoss)
    perfRecord.errorValid.append(validLoss)
    perfRecord.errorTest.append(testLoss)
    
    perfRecord.trainSetAcc.append(trainAccuracy)
    perfRecord.validSetAcc.append(validAccuracy)
    perfRecord.testSetAcc.append(testAccuracy)
    
    print('Epoch is', epoch,'Loss is ', loss, 'Test Acc is ', testAccuracy, 'Train Acc is', trainAccuracy)
    
    
    
    
#%% Part 1 plots

#Accuracy plots
plt.figure(figsize=(15,15))
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title('Training, Valid and Test Set Accuracy, Early stopping with Epochs = %f' % epochs)
plt.plot(perfRecord.trainSetAcc, 'r')
plt.plot(perfRecord.validSetAcc, 'g')
plt.plot(perfRecord.testSetAcc, 'b')
plt.legend(['Training Set, fin acc = %f' % perfRecord.trainSetAcc[-1] 
            , 'Valid Set, fin acc = %f' % perfRecord.validSetAcc[-1] 
            ,'Test Set, fin acc = %f' % perfRecord.testSetAcc[-1], ], loc = 4)


plt.savefig("Accuracy and loss plots 1.4b early stop.png")
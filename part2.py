#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import starter
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


#x = trainData
#y = trainTarget
#reg = 0.1
#W, record = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, 0.001)
#
#
#plt.figure(figsize=(15,15))
#plt.ylabel('Loss')
#plt.xlabel('Iterations')
#plt.title('TrainingSet Error of Different Alpha vs. Iterations of Gradient Descent')
#plt.plot(record.iter, record.loss)


def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


    
def crossEntropyLoss(W, b, x, y, reg):
    def _sigmoid(z):
        return 1/(1+np.exp(-z))
    
    y_predicted = _sigmoid (np.dot(x, W) + b)
    L_D = - np.mean (y * np.log(y_predicted) + (1-y) * np.log(1-y_predicted))
    
    L_W = reg/2 * (np.linalg.norm(W)**2) #Euclidean Norm at 2nd order

    L = L_D + L_W
    return L    
    
def gradCE(W, b, x, y, reg):
    def _sigmoid(z):
        return 1/(1+np.exp(-z))
    
    y_predicted = _sigmoid (np.dot(x, W) + b)
    
    N = len(y)
    grad_W = np.dot(x.T, y_predicted - y) / N + reg*W
    grad_b = np.sum(y_predicted - y)
    return grad_W, grad_b
    
def grad_descent(W, b, x, y, alpha, epochs, reg, EPS, lossType="None"):
    iteration = []
    loss = []
    for t in range(epochs):
        l = crossEntropyLoss(W, b, x, y, reg)
        g_W, g_b = gradCE(W, b, x, y, reg) 
        W = W - alpha*g_W
        b = b - alpha*g_b
        print('iter: %d, loss: %f' % (t, l))
        iteration.append(t)
        loss.append(l)
#        if (np.linalg(g) < EPS): #Euclidean Norm L1 of gradient
#            break
    return W, iteration, loss
    
def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    return

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
    
    L_W = reg/2 * (np.linalg.norm(W)**2)

    L = L_D + L_W
    return L    

def gradMSE(W, b, x, y, reg):
    N = x.shape[0] #number of data vectors we have   
    #Gradient with respect to b
    grad_wrtb = (1/N)*(np.matmul(x,W) + b - y).sum()  #checked with test and is correct
    #Gradient with respect to W, the last term, python adds it to every row, getting the intended effect
    grad_wrtW = (1/N)*(np.matmul(x,W) + b - y)
    grad_wrtW = np.multiply(grad_wrtW,x).sum(axis=0) + reg*W.T 
    grad_wrtW = grad_wrtW.T    

    return grad_wrtW, grad_wrtb

def MSE(W, b, x, y, reg):
    N = x.shape[0] #number of data vectors we have
    meanSquaredError = ((1/(2*N))*(np.linalg.norm((np.matmul(x, W)) + b - y)**2) 
                        + (reg/2)*(np.linalg.norm(W))**2) #I tested with test data and worked

    return meanSquaredError

def gradCE(W, b, x, y, reg):
    def _sigmoid(z):
        return 1/(1+np.exp(-z))
    
    y_predicted = _sigmoid (np.dot(x, W) + b)
    
    N = len(y)
    grad_W = 1/N * np.dot(x.T, y_predicted - y) + reg*W
    grad_b = np.sum(y_predicted - y)
    return grad_W, grad_b
    
def grad_descent(W, b, x, y, alpha, epochs, reg, EPS, lossType="None"):
    loss = []
    accuracy = []
    prev_l = np.inf
    
    def _accuracy(W, b, x, y, lossType):
        if (lossType == 'MSE'):
            y_predicted = np.matmul(x,W) + b
        else:
            y_predicted = 1/(1+np.exp(- (np.dot(x, W) + b)))
        return np.mean(np.round(y_predicted) == y)
    
    for t in range(epochs):
        #pick type of classifier
        acc = _accuracy(W, b, x, y, lossType)
        if (lossType == 'MSE'):    
            l = MSE(W, b, x, y, reg)
            g_W, g_b = gradMSE(W, b, x, y, reg)
        else:
            l = crossEntropyLoss(W, b, x, y, reg)
            g_W, g_b = gradCE(W, b, x, y, reg)
        #keep record of loss
        loss.append(l)
        accuracy.append(acc)
        print('iter: %d, loss: %f, acc: %f' % (t, l, acc))
        #update W & b    
        W = W - alpha*g_W
        b = b - alpha*g_b
        #break if loss difference is smaller than error tolerance
        if (abs(l-prev_l) < EPS):
            break
        #update previous loss
        prev_l = l
        
    return W, b, loss, accuracy
    
def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    return

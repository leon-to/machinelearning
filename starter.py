import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def MSE(W, b, x, y, reg):
    N = x.shape[0] #number of data vectors we have
    meanSquaredError = (1/(2*N))*(np.linalg.norm((np.matmul(x, W)) + b - y)**2 
                        + (reg/2)*(np.linalg.norm(W))**2) #I tested with test data and worked

    return meanSquaredError

    
def gradMSE(W, b, x, y, reg):
    N = x.shape[0] #number of data vectors we have   
    #Gradient with respect to b
    grad_wrtb = (1/N)*(np.matmul(x,W) + b - y).sum()  #checked with test and is correct
    #Gradient with respect to W, the last term, python adds it to every row, getting the intended effect
    grad_wrtW = (1/N)*(np.matmul(x,W) + b - y)
    grad_wrtW = np.multiply(grad_wrtW,x).sum(axis=0) + reg*W.T 
    grad_wrtW = grad_wrtW.T    

    return grad_wrtb, grad_wrtW
    
def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    
    return 
    
def gradCE(W, b, x, y, reg):
    # Your implementation here
    return
    
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg): #version with no EPS, add erorr tol
    for i in range(0, iterations):
        error = MSE(W, b, trainingData, trainingLabels, reg)
        grad_wrtb, grad_wrtW = gradMSE(W, b, trainingData, trainingLabels, reg)
        v_t_W = -grad_wrtW
        v_t_b = -grad_wrtb
        W = W + alpha*v_t_W
        b = b + alpha*v_t_b
        print('iteration = %d error = %f' % (i, error))
    
    return W, b
    
def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    return
    
def classify(W, b, x, y):
    y_hat = np.matmul(x,W) + b
    
    #threshold
    indicesPos = y_hat >= 0.5
    indicesNeg = y_hat < 0.5
    y_hat[indicesPos] = 1
    y_hat[indicesNeg] = 0

    accuracy = np.mean( y_hat == y )
    misclassIndices = y_hat != y
    print('accuracy is %f' % accuracy)
    
    return y_hat, accuracy, misclassIndices
    
    
    

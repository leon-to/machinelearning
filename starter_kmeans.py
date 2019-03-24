import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import time

#%% Function defs and dataObject

class perfData:
    
    def __init__(self):
        self.loss = []
        self.lossValid= []
        self.finalMU = []
        self.finalPWDMat = []

def error(X, MU):
    #From https://databricks.com/tensorflow/training-and-convergence
    pair_dist = distanceFunc(X,MU)
    #Compute min in each row
    pair_dist_min = tf.reduce_min(pair_dist, axis = 1)
    loss = tf.reduce_sum(pair_dist_min)
        
    return loss


# Distance function for K-means
def distanceFunc(X, MU): 
    #From https://databricks.com/tensorflow/training-and-convergence
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(MU, 1)
    pair_dist = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    pair_dist = tf.transpose(pair_dist)
    
    
    return pair_dist
    
#%%

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

is_valid = False
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

trainData = data


#Temporary random points, change this
#Initialize the MU (centers), to be K random points. 
np.random.seed(45689)
[N, dim] = np.shape(trainData)
#==============================================================================
# randID = np.random.randint(N, size = K) #Random points
# initCenters = trainData[randID,:] #CHANGE THIS TO SAPMLING FORM STANDARD NORMAL DIST
# 
#==============================================================================


#%% Part 1 Tensorflow 
K = 3
initCenters = np.random.normal(size = (K, 2))


#Data storage
perfDataObject = perfData()


#TF model
X = tf.placeholder(tf.float64, shape = (N, dim))
MU = tf.Variable(initCenters, name = 'MU')
loss = error(X,MU)
train_op = tf.train.AdamOptimizer(learning_rate = 0.01, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    for i in range(1000):
        _, lossCalc = session.run([train_op, loss], feed_dict = {X : trainData})
        outMU = session.run(MU)
        
        perfDataObject.loss.append(lossCalc)
        
        
        print(lossCalc)
        print(outMU)
        print('epoch = %i' % i)

        
    #Assign to clusters
    pairwiseDistMat = distanceFunc(trainData, outMU)
    arrayPDM = pairwiseDistMat.eval(session = session)
    dataClusterID = arrayPDM.argmin(axis = 1)
    
    plt.figure(figsize=(15,15))
    
    for i in range(0, K): #Plot each cluster
        indicesThisCluster = [index for index, value in enumerate(dataClusterID) if value == i]
        colors = ['r', 'g', 'b', 'c', 'y', 'b']
        plt.scatter(trainData[indicesThisCluster, 0], trainData[indicesThisCluster, 1], color=colors[i])
        #plot the center point
        plt.scatter(outMU[i, 0], outMU[i, 1], marker = 'o', color= 'k', s = 100)         

    plt.title('Scatter plot with different colours per cluster, cluster center in black')        
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.savefig('Q1P1scatter.png') 

    #Plot loss
    plt.figure(figsize=(15,15))
    plt.plot(perfDataObject.loss)    
    plt.title('Loss vs Iter, K = 3')        
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Euclidean Distance)')
    plt.savefig('Q1P1loss.png')           
    
    
        
        

        
#%% Part 2 

Kparams = [1,2,3,4,5]
#K = 3
plt.figure(figsize=(23,23))
for K in Kparams:
    
    plt.subplot(3,2,K)
    np.random.seed(400)
    initCenters = np.random.normal(size = (K, 2))
    
    #Data storage
    perfDataObject = perfData()
    string = [];
    
    
    #TF model
    X = tf.placeholder(tf.float64, shape = (N, dim))
    MU = tf.Variable(initCenters, name = 'MU')
    loss = error(X,MU)
    train_op = tf.train.AdamOptimizer(learning_rate = 0.01, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)
    
    model = tf.global_variables_initializer()
    
    
    percentage = []
    with tf.Session() as session:
        session.run(model)
        for i in range(1000):
            _, lossCalc = session.run([train_op, loss], feed_dict = {X : trainData})
            outMU = session.run(MU)
            
            perfDataObject.loss.append(lossCalc)
            
            print(lossCalc)
            print('epoch = %i K = %i' % (i, K))
    
            
        #Assign to clusters
        pairwiseDistMat = distanceFunc(trainData, outMU)
        arrayPDM = pairwiseDistMat.eval(session = session)
        dataClusterID = arrayPDM.argmin(axis = 1)
        
        
        for i in range(0,K): #Plot each cluster
            indicesThisCluster = [index for index, value in enumerate(dataClusterID) if value == i]
            percentage.append(np.mean(dataClusterID == i)*100)
            colors = ['r', 'g', 'm', 'c', 'y', 'b']
            plt.scatter(trainData[indicesThisCluster, 0], trainData[indicesThisCluster, 1], color=colors[i])
            
         
        #Legend
        if K == 1 :        
            string = 'Clust 1 100 %%' 
        elif K == 2:
            string = ('Clust 1 %.2f %%' % percentage[0],'Clust 2 %.2f %%' % percentage[1])
        elif K == 3:
            string = ('Clust 1 %.2f %%' % percentage[0],'Clust 2 %.2f %%' % percentage[1], 'Clust 3 %.2f %%' % percentage[2])
        elif K == 4:
            string = ('Clust 1 %.2f %%' % percentage[0],'Clust 2 %.2f %%' % percentage[1], 'Clust 3 %.2f %%' % percentage[2], 'Clust 4 %.2f %%' % percentage[3])
        elif K == 5:
            string = ('Clust 1 %.2f %%' % percentage[0],'Clust 2 %.2f %%' % percentage[1], 'Clust 3 %.2f %%' % percentage[2], 'Clust 4 %.2f %%' % percentage[3], 'Clust 5 %.2f %%' % percentage[4] )
                          
        plt.legend(string)
        
        #Plot the centrepoints
        for i in range(0, K):
            plt.scatter(outMU[i, 0], outMU[i, 1], marker = 'o', color= 'k', s = 100) 
    
        plt.title('Scatter plot of clusters, cluster center (black), K = %i' % K)        
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        
plt.savefig('Q1P2.png')



#%% Saved old code
def error(X, MU):
    #From https://databricks.com/tensorflow/training-and-convergence
    pair_dist = distanceFunc(X,MU)
    #Compute min in each row
    pair_dist_min = tf.reduce_min(pair_dist, axis = 1)
    loss = tf.reduce_sum(pair_dist_min)
        
    return loss


# Distance function for K-means
def distanceFunc(X, MU): 
    #From https://databricks.com/tensorflow/training-and-convergence
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(MU, 1)
    pair_dist = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    pair_dist = tf.transpose(pair_dist)
    
    
    return pair_dist
    
    
def error(X, MU):
    #From https://databricks.com/tensorflow/training-and-convergence
    pair_dist = distanceFunc(X,MU)
    #Compute min in each row
    pair_dist_min = tf.reduce_min(pair_dist, axis = 1)
    loss = tf.reduce_sum(pair_dist_min)
        
    return loss


# Distance function for K-means
def distanceFunc(X, MU): 
    #From https://databricks.com/tensorflow/training-and-convergence
    
    firstRun = True
    
    for i in range(0, K): #Loop through each center
        center = MU[i,:]
        #Calculate distance to each datapoint
        dist = tf.norm(tf.subtract(X, center), axis=1)
        dist = tf.reshape(dist, [10000,1])
        #Concat it. 
        if firstRun:
            pair_dist = dist
            firstRun = False
        else:
            pair_dist = tf.concat([pair_dist, dist], 1)
             
    return pair_dist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

#Flags
is_valid = False

#% %# Loading data

data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

#%% Global Parameters
K = 3
trainData = data


#%% Tensorflow Model

#Initialize the MU (centers), to be K random points. 
[N, dim] = np.shape(data)
randID = np.random.randint(N, size=K)
centers = trainData[randID,:]




# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    
    [dimX, _ ] = np.shape(X)
    pair_dist = np.empty((dimX,0), dtype = 'float64')
    
    for center in MU: #Loop through each center
        #Calculate distance to each datapoint
        dist = np.linalg.norm(X - center, axis=1)
        dist = dist.reshape((dimX,1))
        #Concat it. 
        pair_dist = np.hstack((pair_dist, dist))
             
    return pair_dist
    
    
def lossKMeans(X, MU):
    #Pairwise distance matrix calc
    pairwiseDistMat = distanceFunc(X, MU) 
    #Get the minimum of each row (i.e. the minimum center point distance, as per formula)
    pairwiseDistMat = pairwiseDistMat.min(axis = 1)
    #Square each dist
    pairwiseDistMat = pairwiseDistMat**2
    #Sum
    loss = pairwiseDistMat.sum()
    
    return loss
    
    
    
    
    
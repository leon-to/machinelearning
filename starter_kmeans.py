import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

#%%
#Flags
is_valid = False

#% %# Loading data

data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

#==============================================================================
# # For Validation set
# if is_valid:
#   valid_batch = int(num_pts / 3.0)
#   np.random.seed(45689)
#   rnd_idx = np.arange(num_pts)
#   np.random.shuffle(rnd_idx)
#   val_data = data[rnd_idx[:valid_batch]]
#   data = data[rnd_idx[valid_batch:]]
#==============================================================================

#%% Global Parameters
K = 3
trainData = data
LEARNINGRATE = 0.001 #for ADAM
epochs = 2000


#%% Tensorflow Model

#Initialize the MU (centers), to be K random points. 
[N, dim] = np.shape(trainData)
randID = np.random.randint(N, size=K)
initCenters = trainData[randID,:] #CHANGE THIS TO SAPMLING FORM STANDARD NORMAL DIST


X = tf.placeholder("float64", np.shape(trainData))
centers = tf.Variable(initCenters)
error = lossKMeans(X, centers)
train_op = tf.train.AdamOptimizer().minimize(error)


with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    for epo in range(epochs):
        _, loss = session.run([train_op, error], {X: trainData})
        print('loss = %f epochs = %i' % (loss, epo))

        
#%% Function Defs
# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    
    #from https://databricks.com/tensorflow/clustering-and-k-means
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(MU, 1)
    pair_dist = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
             
    return pair_dist
    
    
def lossKMeans(X, MU):
    #Pairwise distance matrix calc
    
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(MU, 1)
    pair_dist = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    
    mins = tf.argmin(pair_dist, 0)
    mins = tf.math.square(mins)
    mins = tf.math.reduce_sum(tf.cast(mins, "float64"))
    return mins















    
    
    
    
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import math


# Distance function for GMM
#def distanceFunc(X, MU):
#    # Inputs
#    # X: is an NxD matrix (N observations and D dimensions)
#    # MU: is an KxD matrix (K means and D dimensions)
#    # Outputs
#    # pair_dist: is the pairwise distance matrix (NxK)
#    
#    pair_dist = tf.reduce_sum(
#        tf.square(
#            tf.subtract( # broadcasting substract NxKxD
#                tf.expand_dims(X, 1), # Nx1xD
#                tf.expand_dims(MU, 0), # 1xKxD
#            )
#        ),
#        2
#    )
#    
#    return pair_dist

# Distance function for K-means
def distanceFunc(X, MU): 
    #From https://databricks.com/tensorflow/training-and-convergence
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(MU, 1)
    pair_dist = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    pair_dist = tf.transpose(pair_dist)
    
    return pair_dist

def log_GaussPDF(x, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    
    # log pdf = -0.5*(x-mu)^2 / sigma^2 - log(2pi*sigma)
#    log_pdf = tf.subtract(
#        -0.5*tf.div(distanceFunc(X, mu), tf.square(tf.transpose(sigma))),
#        tf.log(tf.sqrt(2*math.pi*tf.transpose(sigma)))          
#    )
    sigma = tf.transpose(sigma)
    log_pdf = tf.log(1/(sigma*tf.sqrt(2*math.pi))) - distanceFunc(x, mu)/(2*tf.square(sigma))
    
    return log_pdf

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    
    # posterior = N (x ; µk, σk2) * pi
    # log posterior = log_pdf + log_pi
#    return tf.add(log_PDF, tf.transpose(log_pi))
    log_pi = tf.transpose(log_pi) # [Kx1] -> [1xK]
    
    # [Kx1] + [NxK] - [Nx1]
    return log_pi + log_PDF - tf.expand_dims(hlp.reduce_logsumexp(log_pi + log_PDF), 1)
        
def compute_loss(log_pdf, log_pi):
    # loss = - sum(log(sum(exp(log pi + log N))))
#    loss = -1 * tf.reduce_sum(
#        hlp.reduce_logsumexp(
#            log_posterior(log_GaussPDF(x, mu, sigma), log_pi)
#        )
#    )  
#    log_pdf = log_GaussPDF(x, mu, sigma)
    log_likelihood = tf.reduce_sum(hlp.reduce_logsumexp(tf.transpose(log_pi) + log_pdf, reduction_indices=1),0)
    
    loss = -1 * log_likelihood
    
    return loss
    

    
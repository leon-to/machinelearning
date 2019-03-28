import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import math

# Loading data
#data = np.load('data100D.npy')
data = np.load('data/data2D.npy')
[num_pts, dim] = np.shape(data)

is_valid = True

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    
    pair_dist = tf.reduce_sum(
        tf.subtract( # broadcasting substract NxKxD
            tf.expand_dims(X, 1), # Nx1xD
            tf.expand_dims(MU, 0), # 1xKxD
        ),
        2
    )
    
    return pair_dist

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    
    
    # pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
    # Z = (2 pi sigma**2)**0.5
    
    pair_dist = distanceFunc(X, mu)
    [N, K] = pair_dist.shape
    
    sigma_square_broadcast = tf.broadcast_to(tf.square(tf.transpose(sigma)), [N, K]) # NxK
    
    log_pdf = tf.log( # log pdf
        tf.divide(
            # exp(-0.5 (x - mu)**2 / sigma**2)
            tf.exp( 
                tf.divide(
                    tf.multiply(-0.5, tf.square(pair_dist)),
                    sigma_square_broadcast
                )
            ), 
            # (2 pi sigma**2)**0.5
            tf.sqrt(
                tf.multiply(
                    2*math.pi,
                    sigma_square_broadcast
                )
            )
        )
    )
    
    return log_pdf

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    
    # N (x ; µk, σk2) * pi
    mul = tf.multiply(log_PDF, log_pi)
    
    log_post = tf.divide(mul, hlp.reduce_logsumexp(mul))
    
    return log_post
        
def compute_loss(x, mu, sigma, pi):
    log_pdf = log_GaussPDF(x, mu, sigma)
    [N, K] = log_pdf.shape
    pi_broadcast = tf.broadcast_to(tf.transpose(pi), [N, K])
    
    loss = tf.reduce_prod(
        hlp.reduce_logsumexp(
            tf.multiply(log_pdf, pi_broadcast)
        )
    )
    return -1 * loss    
    

    
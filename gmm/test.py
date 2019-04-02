# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:17:32 2019

@author: khoit
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:34:34 2019

@author: khoit
"""

import starter_gmm as st
import tensorflow as tf
import numpy as np
import helper as hlp
import math
import matplotlib.pyplot as plt

data = np.float32(np.load('../res/data2D.npy'))

plt.scatter(data[:,0], data[:,1])
[N, D] = np.shape(data)

K = 3

tf.reset_default_graph()
#tf.set_random_seed(421)

x = tf.placeholder(tf.float32, shape = (N, D))

# initialization step

pi    = tf.get_variable("pi", dtype=tf.float32, initializer=np.float32(np.full([K,1], 1/K)))

#random_idx = tf.squeeze(tf.multinomial(tf.ones([1, tf.shape(x)[0]]), K))
rand_idx = np.random.randint(0, N-1, [K,D])
mu = tf.Variable(tf.gather_nd(x, np.expand_dims( np.random.randint(0,N-1, K), 1)), dtype=tf.float32)
#mu = tf.get_variable("mu", dtype=tf.float32, initializer=tf.gather_nd(x, [[0],[200],[234]]))
#mu    = tf.get_variable("mu", [K, D], dtype=tf.float32, initializer=tf.random_uniform_initializer)
sigma = tf.get_variable("sigma", [K, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer)

#mu    = tf.random.uniform(shape=[K, D], dtype=tf.float64)
#sigma = tf.random.uniform(shape=[K, 1], dtype=tf.float64)

# constraint
sigma = tf.exp(sigma)
log_pi = hlp.logsoftmax(pi)

# Expectation (E) Step: calculate posterior
pair_dist = st.distanceFunc(x, mu)
log_pdf = st.log_GaussPDF(x, mu, sigma)
log_post = st.log_posterior(log_pdf, log_pi) # [NxK]

# Maximization (M) Step
# mu [K,D] = sum(pi*x) / sum(pi)
pi = tf.reduce_sum(tf.exp(log_post), 0) # [NxK] -T-> [KxN] -> [Kx1]

#ex_1 = tf.expand_dims(log_post,2)
#log_x = tf.log(x)
#ex_2 = tf.expand_dims(log_x,1)
#logsumexp = hlp.reduce_logsumexp(ex_1 + ex_2, 0)
#N_j = tf.expand_dims(hlp.reduce_logsumexp(log_post, reduction_indices=0), 1)
#mu = tf.subtract(
#    logsumexp,
#    tf.expand_dims(hlp.reduce_logsumexp(log_post, reduction_indices=0), 1)
#)
mu = tf.divide(
    tf.reduce_sum(
        tf.multiply(
            tf.expand_dims(tf.exp(log_post), 2),
            tf.expand_dims(x,1)
        ),
        0
    ),
    tf.expand_dims(tf.reduce_sum(tf.exp(log_post), 0), 1)        
)
sigma = tf.exp(
    tf.divide(
        tf.reduce_sum(
            tf.multiply(
                tf.exp(log_post),
                pair_dist
            ),
            0
        ),
        tf.reduce_sum(tf.exp(log_post), 0)        
    )
)
# loss = - sum(log(sum(exp(log pi + log N))))
#loss = -1 * tf.reduce_sum(hlp.reduce_logsumexp(log_pdf + pi))
#loss = st.compute_loss(x, mu, sigma, pi)
loss = st.compute_loss(log_pdf, log_pi)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init, feed_dict = {x : data})
    for i in range(1000):
        mu_val, pi_val, dist_val, pdf_val, loss_val, _ = sess.run(
            [
                mu, 
                pi,
                pair_dist,
                log_pdf,
                loss, 
                optimizer
            ], 
            feed_dict = {x : data}
        )
#        print('Pair dist:', dist_val, 'Min:', np.min(dist_val), 'Max:', np.max(dist_val))
#        print('Pdf:', pdf_val, 'Min:', np.min(pdf_val), 'Max:', np.max(pdf_val))
        print('mu:', mu_val)
        print('pi:', pi_val)
        print('Loss:', loss_val) 
        
        #debug
#        print('Epoch:', i)
#        ex_1_val, log_x_val, ex_2_val, logsumexp_val, N_j_val = sess.run(
#            [ex_1, log_x, ex_2, logsumexp, N_j],
#            feed_dict = {x : data}
#        )
#        print('ex 1', ex_1_val)
#        print('ex 2', ex_2_val)
#        print('log sum exp:', logsumexp_val)
#        print('N j:', N_j_val)
    
        
    for k in range(K):
        plt.scatter(mu_val[k, 0], mu_val[k, 1])
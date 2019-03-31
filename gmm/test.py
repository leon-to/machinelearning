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

data = np.load('../data/data2D.npy')
[N, D] = np.shape(data)

K = 3

tf.reset_default_graph()
#tf.set_random_seed(421)

x = tf.placeholder(tf.float32, shape = (N, D))

mu    = tf.get_variable("mu", [K, D], dtype=tf.float32, initializer=tf.random_uniform_initializer)
sigma = tf.get_variable("sigma", [K, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer)
pi    = tf.get_variable("pi", [K, 1], dtype=tf.float32, initializer=tf.random_uniform_initializer)

# constraint
mu = tf.exp(mu)
log_pi = hlp.logsoftmax(pi)

# distance
pair_dist = tf.reduce_sum(
    tf.square(
        tf.subtract( # broadcasting substract NxKxD
            tf.expand_dims(x, 1), # Nx1xD
            tf.expand_dims(mu, 0), # 1xKxD
        )
    ),
    2
)

# log pdf = -0.5*(x-mu)^2 / sigma^2 - log(2pi*sigma)
log_pdf = tf.subtract(
    -0.5*tf.div(pair_dist, tf.square(tf.transpose(sigma))),
    tf.log(tf.sqrt(2*math.pi*tf.transpose(sigma)))          
)

# log posterior
log_posterior = tf.add(log_pdf, tf.transpose(log_pi))

# loss = - sum(log(sum(exp(log pi + log N))))
loss = -1 * tf.reduce_sum(hlp.reduce_logsumexp(log_posterior))


# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print('mu:', mu.eval())
    print('sigma:', sigma.eval())
    print('pi:', pi.eval())
    print('log pi:', log_pi.eval())
    for i in range(1000):
        dist_val, pdf_val, loss_val, _ = session.run(
            [
                pair_dist,
                log_pdf,
                loss, 
                optimizer
            ], 
            feed_dict = {x : data}
        )
        print('Pair dist:', dist_val, 'Min:', np.min(dist_val), 'Max:', np.max(dist_val))
        print('Pdf:', pdf_val, 'Min:', np.min(pdf_val), 'Max:', np.max(pdf_val))
        print('Loss:', loss_val) 
        
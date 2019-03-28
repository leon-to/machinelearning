# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:34:34 2019

@author: khoit
"""

import starter_gmm as st
import tensorflow as tf
import numpy as np

data = np.load('data/data2D.npy')
[N, D] = np.shape(data)

K = 3

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape = (N, D))
mu = tf.Variable(tf.random_uniform((K, D), dtype=tf.float32))
sigma = tf.Variable(tf.random_uniform((K, 1), dtype=tf.float32))
pi = tf.Variable(tf.random_uniform((K, 1), dtype=tf.float32))

loss = st.compute_loss(x, mu, sigma, pi)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
#    for i in range(1000):
    session.run(optimizer, feed_dict = {x : data})
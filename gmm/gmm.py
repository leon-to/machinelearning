# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:34:34 2019

@author: khoit
"""

import starter_gmm as st
import tensorflow as tf
import numpy as np
import helper as hlp

class GaussianMixtureModel (object):
    def __init__(self, data, recorder):
        self.data = data
        self.recorder = recorder
    
    
    
    def build(self, K=3, learning_rate=0.0001):
        tf.reset_default_graph()
        #tf.set_random_seed(421)
        
        [N, D] = self.data.data.shape
        
        self.x = tf.placeholder(tf.float64, shape = (None, D))
        
        
        np_sigma = np.resize(np.var(self.data.train), [K,1])
        np_pi = np.full([K,1], 1/K)
        
        self.mu    = tf.get_variable("mu", [K, D], dtype=tf.float64, initializer=tf.random_uniform_initializer)
        self.sigma = tf.get_variable("sigma", initializer=np_sigma)
        self.pi    = tf.get_variable("pi", initializer=np_pi)
        
        # constraint
        self.mu = tf.exp(self.mu)
        self.log_pi = hlp.logsoftmax(self.pi)
        
        self.loss = st.compute_loss(self.x, self.mu, self.sigma, self.log_pi)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)



    def train(self, epochs, is_valid=True):
        init = tf.global_variables_initializer()
        
        with tf.Session() as session:
            session.run(init)
            
            for i in range(epochs):
                # train model & retrieve model parameters
                mu_val, sigma_val, log_pi_val, loss_train_val, _ = session.run(
                    [   
                        self.mu, self.sigma, self.log_pi,
                        self.loss, self.optimizer
                    ], 
                    feed_dict = {self.x : self.data.train}
                )
                
                # record training results
                print('Train Loss:', loss_train_val) 
                self.recorder.train = self.recorder.train.append(
                    { 
                        'mu': mu_val,
                        'sigma': sigma_val,
                        'log_pi': log_pi_val,
                        'loss': loss_train_val
                     }, 
                    ignore_index=True
                )
                
                # is validated
                if is_valid:
                    # test on valid data
                    loss_valid_val = session.run(
                        self.loss, 
                        feed_dict = {self.x : self.data.valid}
                    )
                    # record valid result
                    self.recorder.valid = self.recorder.valid.append(
                        {'loss': loss_valid_val},
                        ignore_index=True
                    )
                
        
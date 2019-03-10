# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:41:45 2019

@author: khoito
"""
from cnn import ConvolutionalNeuralNetwork
import tensorflow as tf

class StochasticGradientDescent(object):
    def __init__(self, data, cnn):
        self.dt  = data 
        self.cnn = cnn    
        
    def build_trainer(self, epochs, batch_size):
        dt = self.dt
        cnn = self.cnn
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            
            n = len(dt.y_train_oh)
            
            x_train = dt.x_train
            x_test = dt.x_test
            x_valid = dt.x_valid
            y_train_oh = dt.y_train_oh
            y_test_oh = dt.y_train_oh
            y_valid_oh = dt.y_valid_oh
            
            x = cnn.x
            y = cnn.y
            optimizer = cnn.optimizer
            loss = cnn.ce_loss
            accuracy = cnn.loss
            
            # SGD
            for i in range(epochs):
                #shuffle 
                x_shuffled, y_shuffled = st.shuffle(dt.x_train, dt.y_train_oh)
                #go through all batches
                for j in range(0, n, batch_size):
                    x_batch, y_batch = x_shuffled[j:j+batch_size], y_shuffled[j:j+batch_size]
                    # run optimizer
                    sess.run (optimizer, feed_dict = {x: x_batch, y: y_batch})
                
                loss_train, acc_train = sess.run ([loss, accuracy], feed_dict = {x: x_train, y: y_train_oh})
                loss_valid, acc_valid = sess.run ([loss, accuracy], feed_dict = {x: x_valid, y: y_valid_oh})
                loss_test, acc_test = sess.run ([loss, accuracy], feed_dict = {x: x_test, y: y_test_oh})
                print ("Iteration: ", i, 
                       "Train: ", loss_train, acc_train, ' \ '
                       "Valid: ", loss_valid, acc_valid, ' \ '
                       "Test: ", loss_test, acc_test
                       )
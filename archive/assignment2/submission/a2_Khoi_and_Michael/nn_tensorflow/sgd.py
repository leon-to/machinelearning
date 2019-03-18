# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:41:45 2019

@author: khoito
"""
from cnn import ConvolutionalNeuralNetwork
import tensorflow as tf


class StochasticGradientDescent(object):
    def __init__(self, data, recorder, cnn):
        self.dt  = data 
        self.rc  = recorder
        self.cnn = cnn    
        
    def build_trainer(self, epochs=50, batch_size=32):
        dt = self.dt
        cnn = self.cnn
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            
            n = len(dt.y_train_oh)
            
            x = cnn.x
            y = cnn.y
            
            # SGD
            for i in range(epochs):
                #shuffle 
                x_shuffled, y_shuffled = dt.shuffle(dt.x_train, dt.y_train_oh)
                #go through all batches
                for j in range(0, n, batch_size):
                    x_batch, y_batch = x_shuffled[j:j+batch_size], y_shuffled[j:j+batch_size]
                    # run optimizer
                    sess.run (cnn.optimizer, feed_dict = {x: x_batch, y: y_batch})
                
                loss_train, acc_train = sess.run ([cnn.loss, cnn.accuracy], feed_dict = {x: dt.x_train, y: dt.y_train_oh})
                loss_valid, acc_valid = sess.run ([cnn.loss, cnn.accuracy], feed_dict = {x: dt.x_valid, y: dt.y_valid_oh})
                loss_test, acc_test = sess.run ([cnn.loss, cnn.accuracy], feed_dict = {x: dt.x_test, y: dt.y_test_oh})
                print ("Iteration: ", i, 
                       "Train: ", loss_train, acc_train, ' \ '
                       "Valid: ", loss_valid, acc_valid, ' \ '
                       "Test: ", loss_test, acc_test
                       )
                
                
                self.rc.train = self.rc.train.append({'loss': loss_train, 'accuracy': acc_train}, ignore_index=True)
                self.rc.valid = self.rc.valid.append({'loss': loss_valid, 'accuracy': acc_valid}, ignore_index=True)
                self.rc.test = self.rc.test.append({'loss': loss_test, 'accuracy': acc_test}, ignore_index=True)
                
            
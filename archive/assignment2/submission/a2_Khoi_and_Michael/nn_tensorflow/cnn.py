# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:01:15 2019

@author: khoit

1. Input Layer
2. A 3 × 3 convolutional layer, with 32 filters, using vertical and horizontal strides of 1.
3. ReLU activation
4. A batch normalization layer
5. A 2 × 2 max pooling layer
6. Flatten layer
7. Fully connected layer (with 784 output units, i.e. corresponding to each pixel)
8. ReLU activation
9. Fully connected layer (with 10 output units, i.e. corresponding to each class)
10. Softmax output
11. Cross Entropy loss
"""

import tensorflow as tf



class ConvolutionalNeuralNetwork(object):
    def build_model(self,
            seed=421,#tf seed    
            alpha=10e-4, #learning rate for ADAM optimizer
            with_dropout=False, p=0.9, #dropout 
            with_regularizers=False, beta=0.01 #regulizer
            ):
        #initialize
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        
        
        # label
        self.y = tf.placeholder(tf.float32, [None, 10], 'y')
        
        # 1. input layer (dim: 28x28x1)
        self.x = tf.placeholder(tf.float32, [None, 28, 28], 'x')
        x_reshaped = tf.reshape (self.x, [-1, 28, 28, 1])
        
        # 2. 3 × 3 conv layer, 32 filters, vertical/horizontal strides of 1
        W_conv = tf.get_variable(
                'W_conv', 
                shape=(3,3,1,32), #0,1:filter size, 2: channel, 3: filter numbers
                initializer=tf.contrib.layers.xavier_initializer() #Xavier scheme
                )
        b_conv = tf.get_variable(
                'b_conv', 
                shape=(32), 
                initializer=tf.contrib.layers.xavier_initializer()
                )
        conv_layer = tf.nn.conv2d(
            input=x_reshaped,
            filter=W_conv,
            strides=[1,1,1,1], #0: image number, 1,2:h/v stride, 3: # of channel
            padding='SAME',
            name='conv_layer'
            )
        conv_layer = tf.nn.bias_add(conv_layer, b_conv) #output dim: 28x28x32
                
        # 3. ReLU activation
        relu_conv = tf.nn.relu (conv_layer)
        
        # 4. A batch normalization layer
        mean, variance = tf.nn.moments(relu_conv, axes=[0])
        
        bnorm_layer = tf.nn.batch_normalization(
            relu_conv,
            mean=mean,
            variance=variance,
            offset=None, scale=None,
            variance_epsilon=1e-3
            )
        
        # 5. A 2 × 2 max pooling layer (dim: 14x14x32)
        maxpool2x2_layer = tf.nn.max_pool( 
            bnorm_layer, 
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME'
            )
        
        # 6. Flatten layer
        flatten_layer = tf.reshape(maxpool2x2_layer, [-1, 14*14*32])
        
        # 7. Fully connected layer (with 784 output units, i.e. corresponding to each pixel)
        W_fcl_784 = tf.get_variable(
                'W_fcl_784', 
                shape=(14*14*32, 784), 
                initializer=tf.contrib.layers.xavier_initializer() #Xavier scheme
                )
        b_fcl_784 = tf.get_variable(
                'b_fcl_784', 
                shape=(784), 
                initializer=tf.contrib.layers.xavier_initializer() #Xavier scheme
                )
        fullconn784_layer = tf.add(tf.matmul(flatten_layer, W_fcl_784), b_fcl_784)
        
        #drop out
        if with_dropout:
            fullconn784_layer = tf.nn.dropout(fullconn784_layer, keep_prob=p)
            
        # 8. ReLU activation
        relu_fcl = tf.nn.relu (fullconn784_layer)
        
        # 9. Fully connected layer (with 10 output units, i.e. corresponding to each class)
        W_fcl_10 = tf.get_variable(
                'W_fcl_10', 
                shape=(784, 10), 
                initializer=tf.contrib.layers.xavier_initializer() #Xavier scheme
                )
        b_fcl_10 = tf.get_variable(
                'b_fcl_10', 
                shape=(10),
                initializer=tf.contrib.layers.xavier_initializer() #Xavier scheme
                )
        fullconn10_layer = tf.add(tf.matmul(relu_fcl, W_fcl_10), b_fcl_10)
        
        # 10. Softmax output
        y_hat = tf.nn.softmax(fullconn10_layer)
        
        # 11. Cross Entropy loss
        # normal loss
        self.loss = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=self.y))
        # loss with l2 regularizers
        if with_regularizers:
            regularizers = tf.nn.l2_loss(W_conv) + tf.nn.l2_loss(W_fcl_784) + tf.nn.l2_loss(W_fcl_10)
            self.loss = tf.reduce_mean(self.loss + beta*regularizers)
        
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(self.loss)
        
        # compute accuracy
        correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
                    
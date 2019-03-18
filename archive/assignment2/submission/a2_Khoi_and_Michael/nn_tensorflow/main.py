# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:18:02 2019

@author: khoit
"""

from cnn import ConvolutionalNeuralNetwork
from sgd import StochasticGradientDescent
from data import Data
from recorder import Recorder
from plotter import Plotter

import matplotlib.pyplot as plt

dt = Data()
rc = Recorder()
cnn = ConvolutionalNeuralNetwork()
sgd = StochasticGradientDescent(dt, rc, cnn)
plotter = Plotter(rc)

dt.load('notMNIST.npz')

#%% 2.1 + 2.2 Convolutional Neural Network +  Stochastic Gradient Descent

cnn.build_model(
    seed=421,#tf seed    
    alpha=1e-4, #learning rate for ADAM optimizer
    with_dropout=False, p=0.9, #dropout 
    with_regularizers=False, beta=0.01 #regulizer
)
sgd.build_trainer (
    epochs=50, 
    batch_size=32
)

# plot loss & accuracy
plotter.plot_train_valid_test('img/basic')

#%% 2.3 L2 Decay 0.01

cnn.build_model(
    seed=421,#tf seed    
    alpha=1e-4, #learning rate for ADAM optimizer
    with_dropout=False, p=0.9, #dropout 
    with_regularizers=True, beta=0.01 #regulizer beta: weight decay
)
sgd.build_trainer (
    epochs=50, 
    batch_size=32
)

# plot loss & accuracy
plotter.plot_train_valid_test('img/decay001')

#%% 2.3 L2 Decay 0.1

cnn.build_model(
    seed=421,#tf seed    
    alpha=1e-4, #learning rate for ADAM optimizer
    with_dropout=False, p=0.9, #dropout 
    with_regularizers=True, beta=0.1 #regulizer beta: weight decay
)
sgd.build_trainer (
    epochs=50, 
    batch_size=32
)

# plot loss & accuracy
plotter.plot_train_valid_test('img/decay01')

#%% 2.3 L2 Decay 0.5

cnn.build_model(
    seed=421,#tf seed    
    alpha=1e-4, #learning rate for ADAM optimizer
    with_dropout=False, p=0.9, #dropout 
    with_regularizers=True, beta=0.5 #regulizer beta: weight decay
)
sgd.build_trainer (
    epochs=50, 
    batch_size=32
)

# plot loss & accuracy
plotter.plot_train_valid_test('img/decay05')

#%% 2.3.2 Dropout Decay 0.9

cnn.build_model(
    seed=421,#tf seed    
    alpha=1e-4, #learning rate for ADAM optimizer
    with_dropout=True, p=0.9, #dropout 
    with_regularizers=False, beta=0.5 #regulizer beta: weight decay
)
sgd.build_trainer (
    epochs=50, 
    batch_size=32
)

# plot loss & accuracy
plotter.plot_train_valid_test('img/dropout09')

#%% 2.3.2 Dropout Decay 0.75

cnn.build_model(
    seed=421,#tf seed    
    alpha=1e-4, #learning rate for ADAM optimizer
    with_dropout=True, p=0.75, #dropout 
    with_regularizers=False, beta=0.5 #regulizer beta: weight decay
)
sgd.build_trainer (
    epochs=50, 
    batch_size=32
)

# plot loss & accuracy
plotter.plot_train_valid_test('img/dropout075')

#%% 2.3.2 Dropout Decay 0.5

cnn.build_model(
    seed=421,#tf seed    
    alpha=1e-4, #learning rate for ADAM optimizer
    with_dropout=True, p=0.5, #dropout 
    with_regularizers=False, beta=0.5 #regulizer beta: weight decay
)
sgd.build_trainer (
    epochs=50, 
    batch_size=32
)

# plot loss & accuracy
plotter.plot_train_valid_test('img/dropout05')

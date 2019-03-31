# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:55:46 2019

@author: khoit
"""

from data import Data
from gmm import GaussianMixtureModel
from recorder import Recorder
from plotter import Plotter

data = Data()
recorder = Recorder()
plotter = Plotter(recorder)
gmm = GaussianMixtureModel(data, recorder)

data.load('../data/data2D.npy')

#%% 2.2.1 For the dataset data2D.npy, set K = 3 and report the best model parameters it has learnt.
#   Include a plot of the loss vs the number of updates.


data.shuffle(is_valid=False)
gmm.build(K=3)
gmm.train(epochs=1000, is_valid=False)

plotter.print_final_params()
plotter.plot_train_loss()

#%% 2. Hold out 1/3 of the data for validation and for each value of K = {1, 2, 3, 4, 5}, train a MoG
#   model. For each K, compute and report the loss function for the validation data and explain
#   which value of K is best. Include a 2D scatter plot of data points colored by their cluster
#   assignments.


for K in [1,2,3,4,5]:
    data.shuffle(is_valid=True)
    gmm.build(K)
    gmm.train(epochs=1000, is_valid=True)

plotter.print_final_params()
plotter.plot_train_loss()

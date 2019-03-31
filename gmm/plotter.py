# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 02:52:07 2019

@author: khoit
"""

import numpy as np

class Plotter(object):
    def __init__(self, recorder):
        self.recorder = recorder
    
    def plot_loss(self, df, title):
        ax = df.plot()
        ax.set_title(title)
        ax.set_xlabel('Updates')
        ax.set_ylabel('Loss')
    
    def plot_train_loss(self):
        self.plot_loss(self.recorder.train['loss'], 'Training loss vs. number of updates')
        
    def plot_valid_loss(self):
        self.plot_loss(self.recorder.valid, 'Validation loss vs. number of updates')
        
    def print_final_params(self):
        final_idx = self.recorder.train['mu'].size - 1
        
        print('Mu:\n', self.recorder.train['mu'][final_idx])
        print('Sigma:\n', self.recorder.train['sigma'][final_idx])
        print('Pi:\n', np.exp(self.recorder.train['log_pi'][final_idx]))
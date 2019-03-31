# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:58:48 2019

@author: khoit
"""

import numpy as np

class Data(object):
    def load(self, filename):
        self.data = np.load(filename)
        
    def shuffle(self, is_valid):
        if is_valid:
            [num_pts, dim] = np.shape(self.data)
            
            valid_batch = int(num_pts / 3.0)
            np.random.seed(45689)
            rnd_idx = np.arange(num_pts)
            np.random.shuffle(rnd_idx)
            
            self.valid = self.data[rnd_idx[:valid_batch]]
            self.train = self.data[rnd_idx[valid_batch:]]
        else:
            self.train = self.data
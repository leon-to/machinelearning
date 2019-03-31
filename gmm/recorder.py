# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 00:08:04 2019

@author: khoit
"""

import pandas as pd

class Recorder(object):
    def __init__(self):
        self.train = pd.DataFrame(columns=['mu', 'sigma', 'log_pi', 'loss'])
        self.valid = pd.DataFrame(columns=['loss'])
        self.train.index.name = 'Epochs'
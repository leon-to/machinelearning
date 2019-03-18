# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:57:13 2019

@author: khoit
"""

import pandas as pd

class Recorder(object):
    def __init__(self):
        self.train = pd.DataFrame(columns=['loss', 'accuracy'])
        self.valid = pd.DataFrame(columns=['loss', 'accuracy'])
        self.test = pd.DataFrame(columns=['loss', 'accuracy'])
        
        self.train.index.name = 'epoch'
        self.valid.index.name = 'epoch'
        self.test.index.name  = 'epoch'
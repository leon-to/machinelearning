# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:00:27 2019

@author: khoito
"""

import numpy as np

class Data(object):
    def load(self, file_path):
        x_train, x_valid, x_test, y_train, y_valid, y_test = self.__load(file_path)
        y_train_oh, y_valid_oh, y_test_oh = self.__convertOneHot(y_train, y_valid, y_test)

        self.x_train = x_train
        self.x_valid = x_valid
        self.x_test = x_test
        self.y_train_oh = y_train_oh
        self.y_valid_oh = y_valid_oh
        self.y_test_oh = y_test_oh

    def shuffle(self, trainData, trainTarget):
        np.random.seed(421)
        randIndx = np.arange(len(trainData))
        target = trainTarget
        np.random.shuffle(randIndx)
        data, target = trainData[randIndx], target[randIndx]
        return data, target

    def __load(self, file_path):
        with np.load(file_path) as data:
            Data, Target = data["images"], data["labels"]
            np.random.seed(521)
            randIndx = np.arange(len(Data))
            np.random.shuffle(randIndx)
            Data = Data[randIndx] / 255.0
            Target = Target[randIndx]
            trainData, trainTarget = Data[:10000], Target[:10000]
            validData, validTarget = Data[10000:16000], Target[10000:16000]
            testData, testTarget = Data[16000:], Target[16000:]
        return trainData, validData, testData, trainTarget, validTarget, testTarget

        
    def __convertOneHot(self, trainTarget, validTarget, testTarget):
        newtrain = np.zeros((trainTarget.shape[0], 10))
        newvalid = np.zeros((validTarget.shape[0], 10))
        newtest = np.zeros((testTarget.shape[0], 10))
    
        for item in range(0, trainTarget.shape[0]):
            newtrain[item][trainTarget[item]] = 1
        for item in range(0, validTarget.shape[0]):
            newvalid[item][validTarget[item]] = 1
        for item in range(0, testTarget.shape[0]):
            newtest[item][testTarget[item]] = 1
        return newtrain, newvalid, newtest

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
import numpy as np
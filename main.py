# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:01:14 2019

@author: khoit
"""
import question1
import question2
import question3
import numpy as np
import matplotlib.pyplot as plt


#%% Question 1
#%% part 3
question1.plot_1_3()
#%% part 4
question1.plot_1_4()
#%% part 5
question1.plot_1_5()




#%% Question 2 part 2
question2.plot_2_2()
#%% Question 2 part 2
question2.plot_2_3()



#%% Question 3 part 2
question3.plot_3_1_2()
#%% Question 3 part 3
question3.plot_3_1_3()
#%% Question 3 part 4 Beta1
question3.plot_3_1_4(B1=0.95)
question3.plot_3_1_4(B1=0.99)
#%% Question 3 part 4 Beta2
question3.plot_3_1_4(B2=0.99)
question3.plot_3_1_4(B2=0.9999)
#%% Question 3 part 4 Epsilon
question3.plot_3_1_4(eps=1e-09)
question3.plot_3_1_4(eps=1e-04)



#%% Question 3 part 5.2
question3.plot_3_1_2(lossType)
#%% Question 3 part 5.3
question3.plot_3_1_3(lossType)
#%% Question 3 part 5.4 Beta1
question3.plot_3_1_4(B1=0.95, lossType)
question3.plot_3_1_4(B1=0.99, lossType)
#%% Question 3 part 5.4 Beta2
question3.plot_3_1_4(B2=0.99, lossType)
question3.plot_3_1_4(B2=0.9999, lossType)
#%% Question 3 part 5.4 Epsilon
question3.plot_3_1_4(eps=1e-09, lossType)
question3.plot_3_1_4(eps=1e-04, lossType)
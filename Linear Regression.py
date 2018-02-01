# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:56:47 2017

@author: likhith
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataIn = pd.read_csv('C:\Users\likhith\Downloads\linear_regression_test_data.csv')  #loading the data from the path using pandas 

x = np.array(dataIn['x'])
y = np.array(dataIn['y'])
y_theoretical = np.array(dataIn['y_theoretical'])

x_mean = np.mean(x) #calculating mean of x and y
y_mean = np.mean(y)
 
beta_1= (np.cov(x, y))[0, 1] / np.var(x)   #calculating beta_0 and beta_1
beta_0 = y_mean - beta_1 * x_mean

y_hat = beta_0 + beta_1 * x

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter( dataIn['x'],dataIn['y_theoretical'],color='red')
ax.scatter( dataIn['x'],dataIn['y'],color='blue')
ax.plot(x,y_hat, color='blue')
ax.set_title('y vs x, y-theoretical vs x,PC axis and Linear regression')
#k=5
#ax.plot([0, k*eigenVectors[0,0]], [0, k*eigenVectors[1,0]],color='green', linewidth=3)
#ax.set_ylabel('y and y_theoretical')
#ax.set_xlabel('x')
fig.show()
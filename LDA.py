
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 17:57:10 2017

   
   
   
@author: likhith
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from numpy import linalg as LA


dataIn = pd.read_csv('C:\Users\likhith\Downloads\ML Homework2\SCLC_study_output_filtered_2.csv') #Loading the dataset
dataIn = pd.read_csv(io.StringIO(u""+dataIn.to_csv(header=None,index=False)), header=None) 

dataIn.iloc[0:20:,0] = 1 #class 1
dataIn.iloc[20:41:,0] = 2 #class 2

x= dataIn.iloc[:,1:20].values   #taking all the data in the columns 
y= dataIn.iloc[:,0].values #taking the classes data



mean = []  #calculating mean for the two classes  
for clas in range(1,3):
   mean.append(np.mean(x[y==clas], axis=0))
   print('Mean of class %s: %s' %(clas, mean[clas-1])) #printing mean of the classes
   print ('/n')

m1=mean[0]   #mean of class1 
m2=mean[1]   #mean of class2
diff=m1-m2  #difference between mean1 and mean2

ScatterWithin = np.zeros((19,19))  # scatterwithin
for clas,means in zip(range(1,3), mean):
    scattermatrixforeachclass = np.zeros((19,19))
    for row in x[y == clas]:
        row, means = row.reshape(19,1), means.reshape(19,1) 
        scattermatrixforeachclass += (row-means).dot((row-means).T)  #scattermatrix for each class 
    ScatterWithin += scattermatrixforeachclass        #calculating scatter within by adding matrix for each class
print('Scatter with in matrix :\n', ScatterWithin) #printing scattter within

W= LA.inv(ScatterWithin).dot(diff)
print('w is',W) #calculating W

X=x.dot(W.T)  #calculating X

plt.subplot(2, 1, 1) #plotting Lda
plt.title(' LDA without using Sklearn')
plt.plot(X[0:20],np.zeros(20).reshape(20,1),'ro',X[20:40],np.zeros(20).reshape(20,1),'bo')







































 
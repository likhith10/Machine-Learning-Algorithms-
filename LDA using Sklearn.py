# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 23:07:58 2018

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


##########################################Using Sklearn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

dataIn.iloc[0:20:,0] = 1 #class 1
dataIn.iloc[20:41:,0] = 2 #class 2

x= dataIn.iloc[:,1:20] #taking all the data in the columns 
y= dataIn.iloc[:,0]#classes
Y=y.convert_objects(convert_numeric=True) #converting dtype from object to numerical
LdaUsingSklearn = LDA(n_components=2)
XcalculationusingSklearn = LdaUsingSklearn.fit_transform(x, Y)
plt.subplot(2, 1, 2) #plotting Lda using sklearn
plt.title(' LDA using Sklearn') 
plt.plot(-XcalculationusingSklearn[0:20],np.zeros(20).reshape(20,1),'ro',-XcalculationusingSklearn[20:40],np.zeros(20).reshape(20,1),'bo')
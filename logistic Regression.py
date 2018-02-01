# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:55:02 2017

@author: likhith
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import LeaveOneOut  # used just for leave out cross validation


iris = datasets.load_iris() #loading iris dataset
X_new = iris.data[50:,2:4] #Not Considering setosa 
#X
X=np.array([100,2]) #array of size
X=(X_new-np.min(X_new,axis=0)/np.max(X_new,axis=0)-np.min(X_new,axis=0)) #scaling
splitS = LeaveOneOut()  #LeaveOut cross validation
X1=splitS.get_n_splits(X) #using Leaveout
y = iris.target[50:]
Y=[]
for i in range(0,len(y)):
    if y[i]==1:
        Y.append(0)
    else:
        Y.append(1)
Y=np.array(Y)
x1=np.ones([100,1])
x2=np.hstack([x1,X])
cost=0
mismatch=0

for trnindx, tstindx in splitS.split(x2):       #test train split
    X_train, X_test = x2[trnindx], x2[tstindx]
    y_train, y_test = Y[trnindx], Y[tstindx]   

    A=0.045     #Alpha
    thetha=np.zeros([3,1])

    for i in range(0,100):     #performing logistic Regression 100 times
        M=np.transpose(thetha).dot(X_train.T)
        ht=1/(1+(np.exp(-M)))
        Y1=y_train-ht
        x3=np.transpose(X_train)
        thetha=thetha+(A*(x3.dot(Y1.T)))
        cost+=1
        

    Y_test_model=(1/(1+(np.exp(-(thetha.T.dot(X_test.T))))))    
    
    if Y_test_model>=0.5:
        predicted=1
    else:
        predicted=0
    
    i=0
    if predicted!=y_test:
        mismatch+=1
        i+=1
    else:
        mismatch+=0
        i+=1


print ('Average error rate is ',mismatch)


print('classiﬁcation results obtained using logistic regression are compared with results of classification using ANN')
print('The error rate for both Ann and Logistic regression are 1 and 6 the results nearly similar')
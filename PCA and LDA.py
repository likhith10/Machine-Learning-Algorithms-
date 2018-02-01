# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:36:48 2017

@author: likhith
"""



###### PCA on x and y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataIn = pd.read_csv('C:\Users\likhith\Downloads\Machine Learning Final\dataset_1.csv') #loading the data from the path using pandas 

v1=dataIn['V1'] 
v2=dataIn['V2']

Data =np.column_stack((v1,v2))  #combining x and y to form matrix
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter( v1,v2,color='red')

ax.set_title(' (1) raw V1 vs v2')    #####################(1)
fig.show()

d=Data 

Mean = d.mean(axis=0)
MeanA = np.tile(Mean, reps=(d.shape[0], 1))
MeanB = d - MeanA

data = MeanB #mean

             
covMatrix = np.cov(data, rowvar=False) #calculating covariance matrix 

eigenValues, eigenVectors = np.linalg.eig(covMatrix) #calculating eigen values and vectors of covariance matrix
v = eigenValues.argsort()[::-1]
eigenValues = eigenValues[v]   #eigenvalues
eigenVectors = eigenVectors[:, v] #eigenvectors
a=np.dot(data,eigenVectors)
var_cl1 = np.var(a[1:,0])
var_cl2 = np.var(a[0:,1])
print('variance of 1',var_cl1)
print('variance of 2',var_cl2)
print('eigen Values are',eigenValues)
print('values of variance and eigen values are almost same')  ################################################(7)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.scatter( v1,v2,color='red')
fig = plt.figure()
ax.plot()
ax.set_title(' (2) V1 vs v2 and pc1 axis')
k=-20
ax.plot([0, k*eigenVectors[0,0]], [0, k*eigenVectors[1,0]],color='green', linewidth=3)

ax.set_ylabel('V2')
ax.set_xlabel('V1')
fig.show()
print('Yes ,I can see clear seperation of data in projection of rAW data ON pc1 axis') ##################(2)

ax.scatter( v1,v2,color='red')
fig = plt.figure()
ax.plot()
ax.set_title(' (3) V1 vs v2 and pc1 axis')     ####################################(3)
k=-40
ax.plot([0, k*eigenVectors[0,0]], [0, k*eigenVectors[1,0]],color='green', linewidth=3)

ax.set_ylabel('V2')
ax.set_xlabel('V1')
fig.show()




######LDA###############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from numpy import linalg as LA


dataIn = pd.read_csv('C:\Users\likhith\Downloads\Machine Learning Final\dataset_1.csv') #Loading the dataset
dataIn = pd.read_csv(io.StringIO(u""+dataIn.to_csv(header=None,index=False)), header=None) 


x= dataIn.iloc[:,0:2]  #taking all the data in the columns 
y= dataIn.iloc[:,2:3].values #taking the classes data



mean1 = []  #calculating mean for the two classes  
mean2=[]
mean1=np.mean(x[y==1],axis=0)
mean2=np.mean(x[y==0],axis=0)
diff=mean1-mean2

x.iloc[0,:2].shape


Scatter_With=0
ScatterWithin = np.zeros((2,2))  # scatterwithin
for i in range(len(y)):
    if y[i]==0:
        Scatter_With+=(x.iloc[i,:2].reshape(2,1)-mean2.reshape(2,1)).dot((x.iloc[i,:2].reshape(2,1)-mean2.reshape(2,1)).T)
        
    else:
        Scatter_With+=(x.iloc[i,:2].reshape(2,1)-mean1.reshape(2,1)).dot((x.iloc[i,:2].reshape(2,1)-mean1.reshape(2,1)).T)
      
W= LA.inv(Scatter_With).dot(diff)
print('(4)w is',W) #calculating W      ########################(4)

X=x.dot(W.T) #calculating X
plt.subplot(2, 1, 1) #plotting Lda
plt.title(' LDA')
plt.plot(X[0:30],np.zeros(30).reshape(30,1),'yo',X[30:],np.zeros(30).reshape(30,1),'bo')
plt.show()

print('(5)',X)
var=np.var(X)
print('variance of projection onto W axis:',var)     ################################(8)
fig = plt.figure()
plt.subplot(2, 1, 1) #plotting Lda
plt.title(' (5)LDA showing Raw data Waxis and Y(Lda output)')     #########################(5)
plt.scatter( v1,v2,color='red')
plt.plot(X[0:30],np.zeros(30).reshape(30,1),'ro',X[30:],np.zeros(30).reshape(30,1),'bo')
plt.plot([0,-60*W[0]],[0,-60*W[1]],'g-')
plt.show()

plt.subplot(1,1,1)
plt.title('(6)  Raw data v1,v2, pc1 axis from PCA and w from LDA ')     #######################(6)
plt.scatter( v1,v2,color='red')
k=-40
plt.plot([0, k*eigenVectors[0,0]], [0, k*eigenVectors[1,0]],color='blue', linewidth=3)
#ax.plot([0, k * myPCAResults['loadings'][0, 1]], [0, k * myPCAResults['loadings'][1, 1]],
#           color='green',linewidth=3)
#plt.plot(X[0:30],np.zeros(30).reshape(30,1),'ro',X[30:],np.zeros(30).reshape(30,1),'bo')
plt.plot([0,-60*W[0]],[0,-60*W[1]],'g-')
fig.show()

################################################################(9)
print('Both Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) are linear transformation techniques  commonly used for dimensionality reduction')
print('PCA is to find  principal components that maximizes the variance for the entire dataset')
print('LDA  computes linear discriminants which represents the axes  that maximizes the separation between the two classes')
print('sometimes LDA is performed after PCA for dimensionality reduction')





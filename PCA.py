#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 00:19:39 2017

@author: likhith
"""


import numpy as np





#Reading the dataset file from the computer
data=np.recfromcsv("/Users/likhith/Downloads/dataset_1.csv")
# considering  each coloumn into one variable and one  using  more  variable in which  all three are combined coloumn wise
d1=data["x"]
d2=data["y"]
d3=data['z']
d4 = np.array([d1,d2,d3])
np.var(d1)
np.var(d2)
np.var(d3)
np.cov(d1,d2)
np.cov(d2,d3)

#Calculating  mean of each variable 
mean_x = np.mean(d1)
mean_y = np.mean(d2)
mean_z = np.mean(d3)


#forming  an array from the   means found out from above step 
mean_vect = np.array([[mean_x],[mean_y],[mean_z]])
print('Mean Vector:\n', mean_vect)


#Calculating covariance for each coloumn
cov_matrix = np.cov([d1,d2,d3])
print('Covariance Matrix:\n', cov_matrix)


#Calculating the Eigen values and Eigen vectors by using  covariance matrix
eig_value_cov, eig_vect_cov = np.linalg.eig(cov_matrix)


#making key value eigen  pairs 
eig_pairs = [(np.abs(eig_value_cov[i]), eig_vect_cov[:,i]) for i in range(len(eig_value_cov))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

for i in eig_pairs:
    print(i[0])
 
    
# reducing a 3-dimensional feature space to a 2-dimensional feature subspace 
matrix_r = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
#combined the two eigenvectors with the highest eigenvalues to construct  d×k-dimensional eigenvector matrix R and printing it
print('Matrix R:\n', matrix_r)


#Transforming the given dataset  into  new Eigen Spaces
transformed = matrix_r.T.dot(d4)
transformed
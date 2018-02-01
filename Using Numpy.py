# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:51:10 2017

@author: likhith
"""
import numpy as np
a = np.array([1.0,4.0,5.0,8.0,float])
a
type(a)
z = np.array([5,7,8],float)
z
type(z)
z[2]
z
a=np.array([[1,2,3],[4,5,6]],float)
a
a.shape
a.dtype
len(a)
10 in a
a = np.array(range(10),float)
a
a = a.reshape((5,2))
a.shape
b = a
b
c = a.copy()
c
a = np.array([1.0,4.0,5.0,8.0,float])
a.tolist()
list(a)
s = a.tostring()
np.fromstring(s)
a = np.array([1.0,4.0,5.0,8.0,float])
a.fill(0)
a
a=np.array([[1,2,3],[4,5,6]],float)
a.transpose()
a.flatten()
a=np.array([1,2,3],float)
b=np.array([4,5,6],float)
c=np.array([7,8,9],float)
np.concatenate((a,b,c))
 np.ones((2,3), dtype=float)
np.zeros((7),dtype=int)
np.identity(4,dtype=float)
a=np.array([1,2,3],float)
b=np.array([4,5,6],float)
a+b
a = np.array([[1,2], [4,5]], float)
a = np.array([[3,4], [6,8]], float)
a*b
a = np.array([[1,2], [3,4]], float)
b = np.array([[2,0], [1,3]], float) 
a * b 
a = np.array([1, 4, 9], float) 
np.sqrt(a) 
a = np.array([1, 4, 5], int) 
for x in a:
           print x
a=np.array([[1,2,3],[4,5,6]],float)
for x in a:
           print x
a = np.array([9, 0, 1], float)
a.argmin()
a.argmax()


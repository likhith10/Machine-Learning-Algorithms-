# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:54:27 2017

@author: likhith
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,0.05,0.1],float) #Taking given X

y=np.array([0.01,0.99],float)  #taking given y

thetha=np.random.rand(2,3) #randomizing thetha

thetha1=np.random.rand(2,3) #randomizing thetha1

k=[]
T=[]
T1=[]
i=0
while True:
    A=np.matmul(thetha,x)
    Asigmo=1/(1+np.exp(-A))
    A2=np.array([1.0,Asigmo[0],Asigmo[1]])
  
    A3=np.matmul(thetha1,A2)
    A3sigmo=1/(1+np.exp(-A3))
    
    Cost_function1=0.5*(((y[0]-A3sigmo[0])**2))
    Cost_function2=0.5*((y[1]-A3sigmo[1])**2)
    Cost_functionFinal=Cost_function1+Cost_function2
    #calculating cost Function
    
    Deltaz13=(A3sigmo[0]-0.01)*(A3sigmo[0]*(1-A3sigmo[0]))
    Deltaz23=(A3sigmo[1]-0.99)*(A3sigmo[1]*(1-A3sigmo[1]))
    Delta3=np.vstack([Deltaz13,Deltaz23])
    Deriv3=Delta3.dot(A2.reshape(-1,1).T)
    Deltafor2=(Asigmo[0])*(1-Asigmo[0])
    Deltafor_2=(Asigmo[1])*(1-Asigmo[1])
    D=np.zeros([2,2])
    D[0][0]=Deltafor2
    D[1][1]=Deltafor_2
    Delta2=Delta3.T.dot(thetha1[:,1:].dot(D))
    Deriv2=Delta2.T.dot(x.reshape(-1,1).T)
    
    thetha2=thetha-Deriv2
    thetha3=thetha1-Deriv3
    k.append(Cost_functionFinal)
    T.append(thetha2)
    T1.append(thetha3)
    i=i+1
    thetha=thetha2
    thetha1=thetha3
 
    if(i==10000):
        break;

plt.plot(k)        

weight1 = []
weight2 = []
weight3 = []
weight4 = []
weight5 = []
weight6 = []
weight7 = []
weight8 = []
weight9 = []
weight10 = []
weight11 = []
weight12 = []




for i in range(0,len(T)): #len(w1_list)
    weight1.append(T[i][0][0])
    weight2.append(T[i][0][1])
    weight3.append(T[i][0][2])
    weight4.append(T[i][1][0])
    weight5.append(T[i][1][1])
    weight6.append(T[i][1][2])
    weight7.append(T1[i][0][0])
    weight8.append(T1[i][0][1])
    weight9.append(T1[i][0][2])
    weight10.append(T1[i][1][0])
    weight11.append(T1[i][1][1])
    weight12.append(T1[i][1][2])
fig2 = plt.figure()
ax = fig2.add_subplot(1,1,1)
ax.set_title('Weight1&2 vs Iterations')
ax.plot( weight1, 'r--' )
ax.plot( weight2, 'g--' )
ax.plot( weight3, 'b--' )
ax.plot( weight4, 'y--' )
ax.plot( weight5, 'm--' )
ax.plot( weight6, 'c' )
ax.plot( weight11, 'r' )
ax.plot( weight12, 'g' )
ax.plot( weight10, 'b' )
ax.plot( weight9, 'y' )
ax.plot( weight8, 'm' )
ax.plot( weight7, 'c' )
ax.set_xlabel('iterations')
ax.set_ylabel('weight1&2')
fig2.show()
    


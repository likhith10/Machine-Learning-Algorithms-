# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 02:10:15 2017

@author: likhith
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import LeaveOneOut


iris = datasets.load_iris()
X_new = iris.data[50:,2:4]#Not Considering setosa 
#X
X=np.array([100,2])#array of size
X=(X_new-np.min(X_new,axis=0)/np.max(X_new,axis=0)-np.min(X_new,axis=0))  #scaling
splitS = LeaveOneOut()   #LeaveOut cross validation
X1=splitS.get_n_splits(X)    #using Leaveout
y = iris.target[50:]
Y=[]

def mismatchcal(j,y):
    mismatch=0
    if j>0.5:
        p=1
    else:
        p=0

    if y==p:
       mismatch+=1
    else:
        mismatch+=0

    return(mismatch)


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

k=[]
T=[]
T1=[]
J=[]
i=0

ct =0
for trnindx, tstindx in splitS.split(x2):    #test train split
    X_train, X_test = x2[trnindx], x2[tstindx]
    y_train, y_test = Y[trnindx], Y[tstindx]
    thetha=np.random.rand(2,3) #randomizing thetha
    thetha1=np.random.rand(1,3)#randomizing thetha1

    cost_func=[]
    j=0
    while j==0:

        Deriv21 = np.zeros(shape=thetha.shape)
        Deriv31 = np.zeros(shape=thetha1.shape)

        for i in range(len(X_train)):
            alpha =0.045
            A=np.matmul(thetha,X_train[i].T)
            Asigmo=1/(1+np.exp(-A))
            A2=np.array([1.0,Asigmo[0],Asigmo[1]])

            A3=np.matmul(thetha1,A2)
            A3sigmo=1/(1+np.exp(-A3))

            
            J=1/2*(((y_train[0]-A3sigmo[0])**2))#Calculating cost function
            cost_func.append(J)


            Deltaz13=(A3sigmo-y_train[i])
            
            Delta3= Deltaz13
            Deriv3=Delta3.dot(A2.reshape(-1,1).T)
            Deriv31=Deriv31+Deriv3
            
            Deltafor2=(Asigmo[0])*(1-Asigmo[0])
            Deltafor_2=(Asigmo[1])*(1-Asigmo[1])
            D=np.zeros([2,2])
            D[0][0]=Deltafor2
            D[1][1]=Deltafor_2
            Delta2=Delta3[0]*(thetha1[:,1:].dot(D))
            Deriv2=Delta2.reshape(-1,1).dot(X_train[0].reshape(-1,1).T)
            Deriv21=Deriv21+Deriv2

        

        thetha2=thetha-(alpha*(Deriv21)*(1/len(X_train)))
        thetha3=thetha1-(alpha*(Deriv31)*(1/len(X_train)))

        T.append(thetha2)
        T1.append(thetha3)



        if (np.round(thetha2,2)==np.round(thetha,2)).all() and (np.round(thetha3,2)==np.round(thetha1,2)).all():
            j=j+1
            break;

        else :
            thetha1 = thetha3
            thetha= thetha2

    A=np.matmul(thetha,X_test.T)
    Asigmo=1/(1+np.exp(-A))
    A2=np.array([1.0,Asigmo[0],Asigmo[1]])

    A3=np.matmul(thetha1,A2)
    A3sigmo=1/(1+np.exp(-A3))


    p=mismatchcal(A3sigmo,y_test)
print('error rate is',p)







cost_func[0]

#plt.plot(k)        

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
#    weight10.append(T1[i][1][0])
#    weight11.append(T1[i][1][1])
#   weight12.append(T1[i][1][2])
#fig2 = plt.figure()
#ax = fig2.add_subplot(1,1,1)
#ax.set_title('Weight1&2 vs Iterations')
#ax.plot( weight1, 'r--' )
#ax.plot( weight2, 'g--' )
#ax.plot( weight3, 'b--' )
#ax.plot( weight4, 'y--' )
#ax.plot( weight5, 'm--' )
#ax.plot( weight6, 'c' )
#ax.plot( weight11, 'r' )
#ax.plot( weight12, 'g' )
#ax.plot( weight10, 'b' )
#ax.plot( weight9, 'y' )
#ax.plot( weight8, 'm' )
#ax.plot( weight7, 'c' )
#ax.set_xlabel('iterations')
#ax.set_ylabel('weight1&2')
#fig2.show()


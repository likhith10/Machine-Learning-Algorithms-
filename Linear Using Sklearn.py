# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:55:08 2017

@author: likhith
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes() #loading the diabetes dataset


diabetes_X = diabetes.data[:, 2] #x for linear regression
diabetes_X =diabetes_X .reshape(len(diabetes_X ),1)

# Split x into training/testing sets
diabetes_X_train, diabetes_X_test = train_test_split(diabetes_X, test_size=0.045, random_state = 30)
# Split y target into training/testing sets
diabetes_y_train,diabetes_y_test = train_test_split(diabetes.target, test_size=0.045, random_state = 30)



reg = linear_model.LinearRegression() 

reg.fit(diabetes_X_train, diabetes_y_train) #training using traning sets

diabetes_predictedy = reg.predict(diabetes_X_test) # predictions using the testing set


# Plotting  testing x vs testing y, and the testing x vs predicted y in the same plot.
plt.title(" Plot : testing x vs testing y and the testing x vs predicted y")
plt.scatter(diabetes_X_test, diabetes_y_test,  color='red')
plt.plot(diabetes_X_test, diabetes_predictedy, color='blue')
plt.show()
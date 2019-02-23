#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:09:55 2018

@author: sachet
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
#here x is an object as it contains the various different
#types of datas in the same matrix

#Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X_labelencoder = LabelEncoder()
X[:, 3] = X_labelencoder.fit_transform(X[:, 3])
X_onehotencoder = OneHotEncoder(categorical_features=[3])
X = X_onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set result
y_pred = regressor.predict(X_test)

'''While building this model we used all the independent variables, but there may be cases where
some independent variables are highly statistically determinant than the another variables and
some that are not statistically determinant at all'''

'''so we must find an optimal team of independent variables so that each independent variable of
the team has great impact on the dependent variable profit, that is each independent variable of the
team is a powerful predictor that is statistically significant and definetly has an effect on the
dependent variable profit and this effect can be positive or negative'''

'''In the multiple linear regression equation we have a const. b0 that is not related to any 
independent variable but we can associate a x0 to the b0 where x0 is 1, and the statsmodel library doesnt
take into account this so we should add this to our matrix of independent variable
Thus we will add a column of ones to our matrix of features that will correspond to this x0
equals one associated to our constant b0
and that is how our statsmodel will understand our multiple linear regression model is
y = b0 + x1*b1 + x2*b2 + ... + xn*bn'''

#building optimal model using backward elimination
# Stats model library is used to get the statistical model of our independent variables
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 
'''first we must create a new optimal matrix of feature, this is going to be at the end the matrix
containing the optimal team of independent variables that are only statistically significant for
dependent variable profit'''
#STEP2: fit the full model with all possible predictors 
x_opt = X[:, [0, 1, 2, 3, 4, 5] ]
'''this variable x_opt will contain independent variables that have high impact on the profit'''
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = X[:, [0, 1, 3, 4, 5] ]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())
#
x_opt = X[:, [0, 3, 4, 5] ]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())
#
x_opt = X[:, [0, 3, 5] ]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
print(regressor_OLS.summary())

x_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

'''after declaration we need to fit the ordinary least square algorithm or the
multiple linear regression itself'''
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
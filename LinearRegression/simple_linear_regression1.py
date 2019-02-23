'''simple linear regression can be represented as y = b0 + b1 * x, and this can be represented as a 
sloped line
    Here 'y' is the dependent variable, 'x' is an independent variable that causes the dependent 
    variable to change'''
#Simple Linear Regression
    
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

'''Here X is the matrix of features while y is the vector of dependent variable'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
#Fitting simple linear regression into the Traning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
"""using this line of code(line 34) the regressor object learnt the correlation
between the salary and the year of experience that is present in the dataset"""
#predicting the test set results
y_pred = regressor.predict(X=X_test)
print()
'''here y_test is the real salary of the people in the company while the y_pred is the 
predicted salary of the people in the test set according to their ages by ML model'''
#Visualising the traning set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()
#Visualising the test set result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()
plt.scatter(X_test, y_pred, color = 'red')
plt.scatter(X_train, regressor.predict(X_train), color = 'blue')
'''Simple linear regression class takes care of the fature scaling'''
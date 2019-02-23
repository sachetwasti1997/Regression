'''
simple linear regression - y = b0 + b1 * x1
multiple linear regression - y = b0+ b1*x1 + b2*x2 + ... + bn*xn
polynomial linear regression - y = b0+ b1*x1 + b2*x1^2+...+ bn*x1^n
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2]

#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
X_lin =lin_reg.fit(X, y) 

#Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
'''the poly_reg object that we are going to create will automatically
create a column with constant 1 to include b0 in the equation'''

#Visualising the results LinearRegression
plt.scatter(X, y, color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.show()

#visualising the result PolynomialRegression
X_new = np.arange(1, 10, 0.1)
X_new = X_new.reshape((len(X_new), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_new, lin_reg2.predict(poly_reg.fit_transform(X_new)), color = 'blue')
plt.show()

#predicting the results using polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))

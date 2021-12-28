import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Regression/Polynomial regression")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X,y)

#Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
regressor2 = LinearRegression()
regressor2.fit(X_poly,y)

#Visualizing linear regression result
plt.scatter(X,y,color="r")
plt.plot(X,regressor1.predict(X),color="b")
plt.title("Linear regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#Visualizing polynomial regression result
X_grid = np.arange(min(X),max(X),0.1).reshape((-1,1))
plt.scatter(X,y,color="r")
plt.plot(X_grid,regressor2.predict(poly.fit_transform(X_grid)),color="b")
plt.title("Polynomial regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

 





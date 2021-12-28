import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Regression/SLR")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Split into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 2/3, random_state = 0)

#Fitting SLR to training set
#线性回归package包含了feature scaling
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test response
y_pred = regressor.predict(X_test)

#Visualizing the result
plt.scatter(X_train,y_train,color="r")
plt.plot(X_train,regressor.predict(X_train),color="b")
plt.title("Salary vs Experience (training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()#表示图像代码写完，可以显示出来了

plt.scatter(X_test,y_test,color="r")
plt.plot(X_train,regressor.predict(X_train),color="b")
plt.title("Salary vs Experience (test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

regressor.coef_
regressor.intercept_








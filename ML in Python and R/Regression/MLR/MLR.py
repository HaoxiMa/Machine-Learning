import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Regression/MLR")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_X = LabelEncoder()
X[:,3] = label_X.fit_transform(X[:,3])#convert string to number
Dummy_variables = OneHotEncoder(categorical_features=[3])#分类变量在第3列
X = Dummy_variables.fit_transform(X).toarray()

#Avoid the Dummy variable trap
#分类变量自由度为2，但生成了三个dummy variables,所以要删除一个
X = X[:,1:]

#Split into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.8, random_state = 0)

#Python的线性回归方法中附带了feature scaling

#Fitting MLR to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set
y_pred = regressor.predict(X_test)

#Build the optimal model using Backward model selection
import statsmodels.api as sm
X_train = np.insert(X_train,0,1,axis=1)#做model selection的函数，并没有默认包含intercept，所以我们要手动添加X0=1
#选择显著性为0.05
X_opt = X_train[:,[0,1,2,3,4,5]]
regressor_select = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_select.summary()#看结果删除X2
X_opt = X_train[:,[0,1,3,4,5]]
regressor_select = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_select.summary()#删除X1
X_opt = X_train[:,[0,3,4,5]]
regressor_select = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_select.summary()#删除X2
X_opt = X_train[:,[0,3,5]]
regressor_select = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_select.summary()#删除X2
X_opt = X_train[:,[0,3]]
regressor_select = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_select.summary()












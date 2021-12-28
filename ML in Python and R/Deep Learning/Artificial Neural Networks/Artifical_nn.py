import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Deep Learning/Artificial Neural Networks")

""" Part 1 - Data Preprocessing """

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_X_1 = LabelEncoder()
X[:,1] = label_X_1.fit_transform(X[:,1])#convert string to number
label_X_2 = LabelEncoder()
X[:,2] = label_X_1.fit_transform(X[:,2])#convert string to number
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])], remainder='passthrough')                                         
X = ct.fit_transform(X)
X = X[:,1:]

#Splitting the dataset into the train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.8, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

""" Part2 - Building ANN """
#Import the libraries
import keras
from keras.models import Sequential #Build ANN
from keras.layers import Dense #在NN中添加新的层

#Build the ANN
classifier = Sequential()
 
#Adding the input layer and the first hidden layer
classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform",input_dim=11))
#输入层有11个神经元 hidden layer有6个神经元  relu--Rectifier Function   
#kernel_initializer="uniform"--随机初始化权重 

#Adding the second hidden layer
classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform"))

#Adding the output layer
classifier.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))
#sigmoid--Sigmoid Function
#当输出为3个类别是，y要转化为dummy variable，并且activation = “softmax”

#Compiling the ANN
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#optimizer参数设置最优化是用到的算法，adam是随机梯度下降算法中的一种算法
#loss 用来设置损失函数, 当>3类当时，loss="categorical_crossentropy
#matrics 用来设置模型评估标准,是一个list，可以设置好几个标准

#Fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)
#批量计算，做100期

""" Making the predictions and evaluating the model """

#Predicting the test set results
y_prob = classifier.predict(X_test)
y_pred = [1 if i > 0.5 else 0 for i in y_prob]

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





















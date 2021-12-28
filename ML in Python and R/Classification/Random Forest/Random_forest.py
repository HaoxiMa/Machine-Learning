import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Classification/Random Forest")

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

#Spliting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.75, random_state = 0)

#算法没有用到距离，同时为了能更好更直观的解释Random forest的分类标准，我们不需要做feature scaling
#但这里我们为了画图，就做一下feature scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

#Fitting Random Forest to training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)#10棵决策树 
classifier.fit(X_train,y_train)

#Predicting the test set
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min()-1, X_set[:,0].max()+1, 0.01),
                     np.arange(X_set[:,1].min()-1, X_set[:,1].max()+1, 0.01))
plt.contourf(X1,X2,classifier.predict(np.c_[X1.ravel(),X2.ravel()]).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(("red","green")))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
#enumerate函数用索引和对应的值来循环
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(("orange","blue"))(i), label=j)
plt.title("Classifier (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
#从图像看，依然过度拟合


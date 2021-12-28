"""k-Fold Cross Validation"""

import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Model Selection")

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

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

#Fitting SVM to training set
from sklearn.svm import SVC
classifier = SVC(kernel="rbf",random_state=0)
classifier.fit(X_train,y_train)


#Predicting the test set
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#Applying grid search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{"C":[1,10,100,1000],"kernel":["linear"]},{"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.5,0.1,0.01,0.001]}]
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring="accuracy",cv=10,n_jobs=-1)
#n_jobs=-1当数据集很大的时候，设置这个参数很有用
grid_search = grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
#看完最佳参数以后，将参数的跨度变小，在最佳参数周围调试参数

#Applying k-Fold cross validation
X = scale_X.fit_transform(X)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier,X,y,cv=10)
accuracies.mean()

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


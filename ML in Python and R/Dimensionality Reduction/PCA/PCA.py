import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Dimensionality Reduction/PCA")

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Spliting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.8, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
#n_components参数表示转化为几个特征
pca.fit(X_train)
explained_variance = pca.explained_variance_ratio_
pca = PCA(n_components=2)#根据explained_variance的值来选择
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#Fitting logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
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
             alpha = 0.5, cmap = ListedColormap(("red","green","yellow")))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
#enumerate函数用索引和对应的值来循环
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(("orange","blue","black"))(i), label=j)
plt.title("Classifier (Training set)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

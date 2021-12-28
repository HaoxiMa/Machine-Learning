import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Natural Language Processing")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

""" Clean the texts """
import re

#去除数字和标点
dataset["Review"][0]
review = re.sub('[^a-zA-Z]'," ",dataset["Review"][0])

#变大写为小写
review = review.lower()

#去除所有虚词
import nltk
nltk.download("stopwords")#下载虚词字典
from nltk.corpus import stopwords
review = review.split()
review = [word for word in review if not word in set(stopwords.words("english"))]

#词根化
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]

#连词成句
review = " ".join(review)

#For loop
corpus = []
for i in range(dataset.shape[0]):
    review = re.sub('[^a-zA-Z]'," ",dataset["Review"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)

#创建稀疏矩阵
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)#取前1500个出现次数较多的词
X = cv.fit_transform(corpus).toarray()

""" Fit model (SVM) """
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.8, random_state = 0)

#Fitting SVM to training set
from sklearn.svm import SVC
classifier = SVC(kernel="linear",random_state=0)
classifier.fit(X_train,y_train)


#Predicting the test set
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)










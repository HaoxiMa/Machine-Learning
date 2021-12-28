import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Clustering")

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

#Feature scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X = scale_X.fit_transform(X)

#Using the elbow method to indentify the optimal number of cluseters
from sklearn.cluster import KMeans
Cost = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, init="k-means++", 
                    random_state=0)#n_init随机选10次初始值
    kmeans.fit(X)
    Cost.append(kmeans.inertia_)
plt.plot(range(1,11),Cost)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("Cost")
plt.show()#5个clusters

#Applying the k-means to the mall dataset
kmeans = KMeans(n_clusters=5, max_iter=300, n_init=10, init="k-means++", random_state=0)
y_kmeans = kmeans.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="red",label="Cluster0")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c="blue",label="Cluster1")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c="green",label="Cluster2")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c="cyan",label="Cluster3")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c="magenta",label="Cluster4")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300,c="yellow",
            label="Centroids")
plt.title("Cluster of customers")
plt.xlabel("Income")
plt.ylabel("Spending score")
plt.legend()
plt.show()








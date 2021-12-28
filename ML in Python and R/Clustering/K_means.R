setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Clustering")

dataset<-read.csv("Mall_Customers.csv")
X<-dataset[,c(4,5)]

#feature scaling
maxs <- apply(X, 2, max)
mins <- apply(X, 2, min)
X<- scale(X, center = mins, scale = maxs - mins)


#Using the elbow method to indentify the optimal #clusters
set.seed(6)
Cost<-c()
for (i in 1:10){
  Cost[i]<-kmeans(X,i,iter.max=300,nstart=20)$tot.withinss
}
plot(c(1:10),Cost,type="b",main="The elbow method",xlab = "#Clusters",ylab = "Cost")
#select 5 clusters

#Fitting K-means to the dataset
kmeans<-kmeans(X,5,iter.max = 300,nstart = 10)
y_kmeans<-kmeans$cluster

#Visualizing the clusters
library(cluster)
clusplot(X,y_kmeans,lines = 0,shade = TRUE,color = TRUE,labels = 2,plotchar = FALSE,span = TRUE,
         main="Clusters of customers",xlab = "Income",ylab = "Spending score")








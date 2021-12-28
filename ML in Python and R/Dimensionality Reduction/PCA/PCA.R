setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Dimensionality Reduction/PCA")

dataset<-read.csv("Wine.csv")

library(caTools)
set.seed(123)
Split<-sample.split(dataset$Customer_Segment,SplitRatio = 0.8)
training_set<-subset(dataset,Split == TRUE)
test_set<-subset(dataset,Split ==FALSE)

#Feature Scaling
maxs <- apply(training_set[,c(1:13)], 2, max)
mins <- apply(training_set[,c(1:13)], 2, min)
training_set[,c(1:13)]<- scale(training_set[,c(1:13)], center = mins, scale = maxs - mins)
test_set[,c(1:13)]<- scale(test_set[,c(1:13)], center =  mins, scale = maxs - mins)

#Applying PCA
library(caret)
library(e1071)
pca<-preProcess(training_set[,-14],method = "pca",pcaComp = 2)#thresh参数用来控制选取方法的百分比
training_set_pca<-predict(pca,training_set)
training_set<-training_set_pca[,c(2,3,1)]
test_set_pca<-predict(pca,test_set)
test_set<-test_set_pca[,c(2,3,1)]

#Fitting SVM to training set
classifier<-svm(Customer_Segment~.,data=training_set,type="C-classification",kernel="radial")
summary(classifier)

#Predicting the test set results 
y_pred<-predict(classifier,test_set[,-3])

#Making confusion matrix
cm <-table(test_set$Customer_Segment,y_pred)

#Visualizing the training set results
set<-training_set
X1<-seq(min(set[,1])-0.1, max(set[,1])+0.1, 0.01)
X2<-seq(min(set[,2])-0.1, max(set[,2])+0.1, 0.01)
grid_set<-expand.grid(X1,X2)
names(grid_set)<-c("PC1","PC2")
y_grid<-predict(classifier,grid_set)
plot(set[,-3],main="CLassifier (Training set)",xlab="PC1",ylab="PC2",xlim=range(X1),ylim=range(X2))
contour(X1,X2,matrix(as.numeric(y_grid),length(X1),length(X2)),add=TRUE)
points(grid_set,col=ifelse(y_grid==2,"Deepskyblue",ifelse(y_grid==1,"red","green")))
points(set[,-3],pch=21,bg=ifelse(set[,3]==2,"black",ifelse(set[,3]==1,"orange","blue")))


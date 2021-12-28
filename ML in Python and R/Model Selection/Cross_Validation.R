setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Model Selection")

dataset<-read.csv("Social_Network_Ads.csv")
dataset<-dataset[,3:5]

library(caTools)
set.seed(123)
Split<-sample.split(dataset$Purchased,SplitRatio = 0.75)
training_set<-subset(dataset,Split == TRUE)
test_set<-subset(dataset,Split ==FALSE)

#Feature Scaling
maxs <- apply(training_set[,c(1,2)], 2, max)
mins <- apply(training_set[,c(1,2)], 2, min)
training_set[,c(1,2)]<- scale(training_set[,c(1,2)], center = mins, scale = maxs - mins)
test_set[,c(1,2)]<- scale(test_set[,c(1,2)], center =  mins, scale = maxs - mins)

#Fitting SVM to training set
library(e1071)
classifier<-svm(Purchased~.,data=training_set,type="C-classification",kernel="radial")
summary(classifier)

#Predicting the test set results 
y_pred<-predict(classifier,test_set[,-3])

#Making confusion matrix
cm <-table(test_set$Purchased,y_pred)

#Applying grid search to find the best parameters
library(caret)
training_set$Purchased<-factor(training_set$Purchased)
grid_search<-train(Purchased~.,data=training_set,method="svmRadial")
grid_search
grid_search$bestTune

#Applying k-Fold cross validation
library(caret)
dataset[,c(1,2)]<- scale(dataset[,c(1,2)], center = mins, scale = maxs - mins)
folds<-createFolds(dataset$Purchased,k=10)#把数据分成10块
cv<-lapply(folds,function(x){
  training_fold<-dataset[-x,]
  test_fold<-dataset[x,]
  classifier<-svm(Purchased~.,data=training_fold,type="C-classification",kernel="radial")
  y_pred<-predict(classifier,test_fold[,-3])
  cm <-table(test_fold$Purchased,y_pred)
  accuracy<-(cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
  return(accuracy)
})
mean(as.numeric(cv))

#Visualizing the training set results
set<-training_set
X1<-seq(min(set[,1])-0.1, max(set[,1])+0.1, 0.01)
X2<-seq(min(set[,2])-0.1, max(set[,2])+0.1, 0.01)
grid_set<-expand.grid(X1,X2)
names(grid_set)<-c("Age","EstimatedSalary")
y_grid<-predict(classifier,grid_set)
plot(set[,-3],main="CLassifier (Training set)",xlab="Age",ylab="EstimatedSalary",xlim=range(X1),ylim=range(X2))
contour(X1,X2,matrix(as.numeric(y_grid),length(X1),length(X2)),add=TRUE)
points(grid_set,col=ifelse(y_grid==1,"red","green"))
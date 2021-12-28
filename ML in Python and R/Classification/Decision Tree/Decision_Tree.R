setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Classification/Decision Tree")

dataset<-read.csv("Social_Network_Ads.csv")
dataset<-dataset[,3:5]

library(caTools)
set.seed(123)
Split<-sample.split(dataset$Purchased,SplitRatio = 0.75)
training_set<-subset(dataset,Split == TRUE)
test_set<-subset(dataset,Split ==FALSE)

#算法没有用到距离，同时为了能更好更直观的解释决策树的分类标准，我们不需要做feature scaling
#但这里我们为了画图，就做一下feature scaling
maxs <- apply(training_set[,c(1,2)], 2, max)
mins <- apply(training_set[,c(1,2)], 2, min)
training_set[,c(1,2)]<- scale(training_set[,c(1,2)], center = mins, scale = maxs - mins)
test_set[,c(1,2)]<- scale(test_set[,c(1,2)], center =  mins, scale = maxs - mins)

#Fitting Decision Tree to training set
library(rpart)
#和Naive Bayes一样需要手动将response转换成分类数据
str(training_set)
training_set$Purchased<-as.factor(training_set$Purchased)
classifier<-rpart(Purchased~., data=training_set)

#在不做feature scaling时，画出决策树
#plot(classifier)
#text(classifier)

#Predicting the test set results 
y_pred<-predict(classifier,test_set[,-3],type = "class")

#Making confusion matrix
cm <-table(test_set$Purchased,y_pred)

#Visualizing the training set results
set<-training_set
X1<-seq(min(set[,1])-0.1, max(set[,1])+0.1, 0.01)
X2<-seq(min(set[,2])-0.1, max(set[,2])+0.1, 0.01)
grid_set<-expand.grid(X1,X2)
names(grid_set)<-c("Age","EstimatedSalary")
y_grid<-predict(classifier,grid_set,type = "class")
plot(set[,-3],main="CLassifier (Training set)",xlab="Age",ylab="EstimatedSalary",xlim=range(X1),ylim=range(X2))
contour(X1,X2,matrix(as.numeric(y_grid),length(X1),length(X2)),add=TRUE)
points(grid_set,col=ifelse(y_grid==1,"red","green"))
points(set[,-3],pch=21,bg=ifelse(set[,3]==1,"orange","blue"))
#比起python，并没有过度拟合



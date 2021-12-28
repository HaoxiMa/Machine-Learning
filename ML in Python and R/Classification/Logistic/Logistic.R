setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Classification/Logistic")

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

#Fitting logistic regression to training set
classifier<-glm(Purchased~.,family=binomial,data=training_set)
summary(classifier)

#Predicting the test set results 
prob_pred<-predict(classifier,type="response",test_set[-3])
y_pred<-ifelse(prob_pred>0.5,1,0)

#Making confusion matrix
cm <-table(test_set$Purchased,y_pred)

#Visualizing the training set results
set<-training_set
X1<-seq(min(set[,1])-0.1, max(set[,1])+0.1, 0.01)
X2<-seq(min(set[,2])-0.1, max(set[,2])+0.1, 0.01)
grid_set<-expand.grid(X1,X2)
names(grid_set)<-c("Age","EstimatedSalary")
prob_set<-predict(classifier,type = "response",grid_set)
y_grid<-ifelse(prob_set>0.5,1,0)
plot(set[,-3],main="CLassifier (Training set)",xlab="Age",ylab="EstimatedSalary",xlim=range(X1),ylim=range(X2))
contour(X1,X2,matrix(as.numeric(y_grid),length(X1),length(X2)),add=TRUE)
points(grid_set,col=ifelse(y_grid==1,"red","green"))
points(set[,-3],pch=21,bg=ifelse(set[,3]==1,"orange","blue"))












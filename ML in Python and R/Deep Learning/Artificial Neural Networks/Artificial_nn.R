setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Deep Learning/Artificial Neural Networks")

#Importing the dataset
dataset<-read.csv("Churn_Modelling.csv")
dataset<-dataset[,4:14]

#Encoding categorical data
str(dataset)
dataset$Geography<-factor(dataset$Geography)
dataset$Gender<-factor(dataset$Gender)

#Splitting the dataset into train and test set
library(caTools)
set.seed(123)
Split<-sample.split(dataset$Exited,SplitRatio = 0.8)
training_set<-subset(dataset,Split == TRUE)
test_set<-subset(dataset,Split ==FALSE)

#Feature Scaling
maxs <- apply(training_set[,-c(2,3,11)], 2, max)
mins <- apply(training_set[,-c(2,3,11)], 2, min)
training_set[,-c(2,3,11)]<- scale(training_set[,-c(2,3,11)], center = mins, scale = maxs - mins)
test_set[,-c(2,3,11)]<- scale(test_set[,-c(2,3,11)], center = mins, scale = maxs - mins) 

#Build ANN for training set
library(h2o)#会用到并行计算
h2o.init(nthreads = -1 )#Connect to H20 并且用所有的计算机核并行计算
classifier<-h2o.deeplearning(y="Exited",training_frame = as.h2o(training_set),activation = "Rectifier",
                             hidden = c(6,6),epochs = 100,train_samples_per_iteration = -2)

#Predicting the test set results 
y_prob<-h2o.predict(classifier,newdata=as.h2o(test_set))
y_pred<-as.vector(y_prob)
y_pred<-ifelse(y_pred>0.5,1,0)

#Making confusion matrix
cm <-table(test_set$Exited,y_pred)

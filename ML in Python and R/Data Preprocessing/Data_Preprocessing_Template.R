###Data Preprocessing Template
setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Data Preprocessing")

#Importing the dataset
dataset<-read.csv("Data.csv")

#Missing value
dataset$Age[is.na(dataset$Age)]<-mean(dataset$Age,na.rm = TRUE)
dataset$Salary[is.na(dataset$Salary)]<-mean(dataset$Salary,na.rm = TRUE)

#Encoding categorical data
dataset$Country<-factor(dataset$Country)
dataset$Purchased<-factor(dataset$Purchased)

#Splitting the dataset into train and test set
library(caTools)
set.seed(123)
Split<-sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set<-subset(dataset,Split == TRUE)
test_set<-subset(dataset,Split ==FALSE)

#Feature Scaling
maxs <- apply(training_set[,c(2,3)], 2, max)
mins <- apply(training_set[,c(2,3)], 2, min)
training_set[,c(2,3)]<- scale(training_set[,c(2,3)], center = mins, scale = maxs - mins)
test_set[,c(2,3)]<- scale(test_set[,c(2,3)], center = mins, scale = maxs - mins) 


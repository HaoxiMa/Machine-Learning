setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Regression/MLR")

#Import dataset
dataset = read.csv("50_Startups.csv")

#Split dataset into train and test set
library(caTools)
set.seed(123)
Split<-sample.split(dataset$Profit,SplitRatio = 0.8)
training_set<-subset(dataset,Split == TRUE)
test_set<-subset(dataset,Split ==FALSE)

#Fitting MLR to training set
fit<-lm(Profit~R.D.Spend+Administration+Marketing.Spend+State,data = training_set)
summary(fit)

#Predicting the test set
y_pred<-predict(fit,test_set)


setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Regression/SLR")

#Import dataset
dataset = read.csv("Salary_Data.csv")

#Split dataset into train and test set
library(caTools)
set.seed(123)
Split<-sample.split(dataset$Salary,SplitRatio = 2/3)
training_set<-subset(dataset,Split == TRUE)
test_set<-subset(dataset,Split ==FALSE)

#Fitting SLR to training set
regressor<-lm(Salary~YearsExperience,data=training_set)
summary(regressor)

#Predict the test 
y_pred<-predict(regressor,test_set)

#Visualizing the training set result
library(ggplot2)
ggplot()+geom_point(aes(x=training_set$YearsExperience,y=training_set$Salary),colour="red")+
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,training_set)),colour="blue",size=1)+
  ggtitle("Salary vs Experience (Training set)")+xlab("Years of experience")+ylab("Salary")

ggplot()+geom_point(aes(x=test_set$YearsExperience,y=test_set$Salary),colour="red")+
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,training_set)),colour="blue",size=1)+
  ggtitle("Salary vs Experience (Test set)")+xlab("Years of experience")+ylab("Salary")

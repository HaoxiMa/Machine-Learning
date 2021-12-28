setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Regression/Polynomial Regression")

#Import dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

#Fitting linear regression 
fit1<-lm(Salary~.,data=dataset)
summary(fit1)

#Fitting polynomial regression
dataset$Level2<-dataset$Level^2
fit2<-lm(Salary~.,data=dataset)
summary(fit2)

#Visualizing the linear regression
library(ggplot2)
ggplot()+geom_point(aes(x=dataset$Level,y=dataset$Salary),colour="red")+
  geom_line(aes(x=dataset$Level,y=predict(fit1,dataset)),colour="blue",size=1)+xlab("Level")+ylab("Salary")

#Visualizing the polynomial regression
X_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot()+geom_point(aes(x=dataset$Level,y=dataset$Salary),colour="red")+
  geom_line(aes(x=X_grid,y=predict(fit2,data.frame(Level=X_grid,Level2=X_grid^2))),colour="blue",size=1)+
              xlab("Level")+ylab("Salary")



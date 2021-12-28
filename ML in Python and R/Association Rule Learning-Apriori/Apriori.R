setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Association Rule Learning-Apriori")

dataset<-read.csv("Market_Basket_Optimisation.csv",header = FALSE)
#已知一共120种产品，数据包含了一周内完成的所有交易的物品数据，一共7501次交易

#创建稀缺矩阵
library(arules)
dataset<-read.transactions("Market_Basket_Optimisation.csv",sep=",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset,topN=50)#画出出现频率最高的前50个物品

#Training Aprior on the dataset
rules<-apriori(dataset,parameter = list(support=0.003,confidence=0.2))
#设定最小的support和confidence:我们希望我们要考虑的物品每天至少卖出去3次，那么一周就是21次，占比为21/7501

#Visualizing the results
inspect(sort(rules,by="lift")[1:10])

setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Natural Language Processing")

#目标，通过对一个餐厅的评价，来判断顾客是否喜欢该餐厅

#文件为tab间隔符文件，不用csv，因为评价中有可能就带有逗号
dataset<-read.delim("Restaurant_Reviews.tsv",quote="",stringsAsFactors = FALSE)#quote=""--省略引号

#######Cleaning the texts
library(NLP)
library(tm)
#创建词袋
corpus<-VCorpus(VectorSource(dataset$Review))
as.character(corpus[[1]])

#变大写字母为小写字母
corpus<-tm_map(corpus,content_transformer(tolower))
as.character(corpus[[1]])

#去除文本中的数字
as.character(corpus[[841]])
corpus<-tm_map(corpus,removeNumbers)
as.character(corpus[[841]])

#去除标点符号
corpus<-tm_map(corpus,removePunctuation)
as.character(corpus[[1]])

#去除虚词
library(SnowballC)
corpus<-tm_map(corpus,removeWords,stopwords())
as.character(corpus[[1]])

#获得每个单词的词根(如：去掉单词的时态等)
corpus<-tm_map(corpus,stemDocument)
as.character(corpus[[1]])

#去除多余空格
corpus<-tm_map(corpus,stripWhitespace)
as.character(corpus[[1]])
 
#将词袋转化为庞大的稀疏矩阵
dtm<-DocumentTermMatrix(corpus)
dtm
#看到稀疏性实在过高，我们把出现次数特别低的单词从矩阵中移除
dtm<-removeSparseTerms(dtm,0.999)#把在行中只出现一次的列删去
dtm

######Fitting classifier (Random forest)
data<-as.data.frame(as.matrix(dtm))
data$Liked<-dataset$Liked

#spliting
library(caTools)
set.seed(123)
Split<-sample.split(data$Liked,SplitRatio = 0.8)
training_set<-subset(data,Split == TRUE)
test_set<-subset(data,Split ==FALSE)

#Fit the model
library(randomForest)
#和Naive Bayes一样需要手动将response转换成分类数据
str(training_set)
training_set$Liked<-as.factor(training_set$Liked)
classifier<-randomForest(x = training_set[,-692],y = training_set$Liked,ntree=10)

#Predicting the test set results 
y_pred<-predict(classifier,test_set[,-692])

#Making confusion matrix
cm <-table(test_set$Liked,y_pred)
cm








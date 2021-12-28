import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Association Rule Learning-Apriori")

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Apriori on dataset
from apyori import apriori
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
#min_length=2:一个rule中必须包含至少两个商品
#要求每天至少卖出3次,则我们取support=3*7/7501
    
#Viusalizing the results
results = list(rules)#不需要像R一样手动排序，已经排好了
myResults = [list(x) for x in results]
myResults[0:10]

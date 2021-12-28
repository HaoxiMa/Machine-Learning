import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Reinforcement Learning/Thompson_Sampling")

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")
#这是一个模拟文件，用来测试算法，数值为1代表用户点开了对应广告。

# Implementing Random Selection
#当我们随机的向用户发送广告时，我们得到的总奖励
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)#0-9中随机生成一个整数
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

#Implementing Thompson Sampling
import random
def Thompson_Sampling_implement(dataset,N,d):#N--用户个数，d--广告个数
    numbers_of_reward_1 = [0] * d
    numbers_of_reward_0 = [0] * d
    ads_selected = []
    for i in range(0,N):
        ad = 0
        max_random = 0
        for j in range(0,d):
            random_beta = random.betavariate(numbers_of_reward_1[j]+1 ,numbers_of_reward_0[j]+1)
            if random_beta > max_random:
                max_random = random_beta
                ad = j
        ads_selected.append(ad)
        reward_i = dataset.values[i,ad]
        if reward_i == 1:
            numbers_of_reward_1[ad] =numbers_of_reward_1[ad] + 1
        else:
            numbers_of_reward_0[ad] =numbers_of_reward_0[ad] + 1
    return np.sum(numbers_of_reward_1),ads_selected


total_reward_new,final_ads_selected = Thompson_Sampling_implement(dataset,dataset.shape[0],dataset.shape[1])
total_reward_new
#可以看出，total reward increase significantly compared to the random selection
plt.hist(final_ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
#应该选择第5个广告
#而且，从图中可以看出，广告5被应用了很多很多次，所以Thompson sampling可以较快的选择出最好的广告


 










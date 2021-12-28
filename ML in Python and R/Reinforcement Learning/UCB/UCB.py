import os
os.chdir("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Reinforcement Learning/UCB")

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

#Implementing UCB
import math
def UCB_implement(dataset,N,d):#N--用户个数，d--广告个数
    Numbers_of_selection = [0]*d
    sums_of_rewards = [0]*d
    ads_selected = []
    for i in range(0,N):
        ad = 0
        max_upper_bound = 0
        for j in range(0,d):
            if (Numbers_of_selection[j]>0):
                average_reward_j = sums_of_rewards[j]/Numbers_of_selection[j]
                delta_j = math.sqrt(3/2 * math.log(i+1)/Numbers_of_selection[j])
                upper_bound_j = average_reward_j + delta_j
            else:
                upper_bound_j =1e400
            if upper_bound_j > max_upper_bound:
                max_upper_bound = upper_bound_j
                ad = j
        ads_selected.append(ad)
        reward_i = dataset.values[i,ad]
        Numbers_of_selection[ad] = Numbers_of_selection[ad]+1
        sums_of_rewards[ad] = sums_of_rewards[ad] + reward_i
    return np.sum(sums_of_rewards),ads_selected


total_reward_new,final_ads_selected = UCB_implement(dataset,dataset.shape[0],dataset.shape[1])
total_reward_new
#可以看出，total reward increase significantly compared to the random selection
plt.hist(final_ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
#应该选择第5个广告

















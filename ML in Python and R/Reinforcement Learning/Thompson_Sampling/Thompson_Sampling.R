setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Reinforcement Learning/Thompson_Sampling")

# Importing the dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
N = 10000
d = 10
ads_selected = integer(0)
total_reward = 0
for (n in 1:N) {
  ad = sample(1:10, 1)
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  total_reward = total_reward + reward
}

# Visualising the results
hist(ads_selected,col = 'blue',main = 'Histogram of ads selections',xlab = 'Ads',ylab = 'Number of times each ad was selected')

#Implementing Thompson Sampling
d = 10
N = 10000
numbers_of_rewards_1 = integer(d)
numbers_of_rewards_0 = integer(d)
ads_selected = c()
total_reward = 0
for (i in 1:N){
  max_random = 0
  ad = 0 
  for (j in 1:d){
    beta_random = rbeta(1,numbers_of_rewards_1[j]+1,numbers_of_rewards_0[j]+1)
    if (beta_random > max_random){
      max_random = beta_random
      ad = j
    }
  } 
  ads_selected = append(ads_selected,ad)
  reward = dataset[i,ad]
  if (reward ==1){
    numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad]+1
    total_reward = total_reward + reward
  }
  else{
    numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad]+1
  }
}

hist(ads_selected)

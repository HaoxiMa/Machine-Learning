setwd("/Users/mahaoxi/Desktop/Machine Learning/ML in Python and R/Reinforcement Learning/UCB")

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

#Implementing UCB
d = 10
N = 10000
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
ads_selected = c()
total_reward = 0
for (i in 1:N){
  max_upper_bound = 0
  ad = 0 
  for (j in 1:d){
    if (numbers_of_selections[j] > 0){
      average_reward = sums_of_rewards[j]/numbers_of_selections[j]
      delta_i = sqrt(3/2 * log(i)/numbers_of_selections[j])
      upper_bound = average_reward +delta_i
    }
    else{
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound){
      max_upper_bound = upper_bound
      ad = j
    }
  }
  ads_selected = append(ads_selected,ad)
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  reward = dataset[i,ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

hist(ads_selected)











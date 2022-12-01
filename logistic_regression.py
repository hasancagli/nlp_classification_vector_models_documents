import nltk                         # NLP toolbox
from os import getcwd
import pandas as pd                 # Library for Dataframes 
from nltk.corpus import twitter_samples 
import matplotlib.pyplot as plt     # Library for visualization
import numpy as np                  # Library for math functions
from utils import process_tweet, build_freqs # Our functions for NLP

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
tweets = all_positive_tweets + all_negative_tweets ## Concatenate the lists. 
labels = np.append(np.ones((len(all_positive_tweets),1)), np.zeros((len(all_negative_tweets),1)), axis = 0)
# split the data into two pieces, one for training and one for testing (validation set) 
train_pos  = all_positive_tweets[:4000]
train_neg  = all_negative_tweets[:4000]
# concatenate pos and neg.
train_x = train_pos + train_neg 
print("Number of tweets: ", len(train_x))

data = pd.read_csv('./data/logistic_features.csv'); # Load a 3 columns csv file using pandas function
print(data.head(10)) # Print the first 10 data entries

# Each feature is labeled as bias, positive and negative
X = data[['bias', 'positive', 'negative']].values # Get only the numerical values of the dataframe
Y = data['sentiment'].values; # Put in Y the corresponding labels or sentiments
print(X.shape) # Print the shape of the X part
print(X) # Print some rows of X
print(Y)

# trained logistic regression model
theta = [6.03518871e-08, 5.38184972e-04, -5.58300168e-04]

# Plot the samples in a scatter plot
fig,ax = plt.subplots(figsize=(8,8))
colors = ["red", "green"]
ax.scatter(X[:,1], X[:,2], c=[colors[int(k)] for k in Y], s = 1)
plt.xlabel("Positive")
plt.ylabel("Negative")
plt.show()
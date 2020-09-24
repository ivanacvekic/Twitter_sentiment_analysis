#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
from wordcloud import WordCloud
import string
string.punctuation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
#from twitterscraper import query_tweets
#import datetime as dt

# uploading the file
tweet_df = pd.read_csv('twitter.csv')
tweet_df

# information about variable type
tweet_df.info()
# column count, mean, sd, min, max, etc. 
tweet_df.describe()
# looking up the 'tweet' column
tweet_df['tweet']

# dropping the 'id' column
tweet_df = tweet_df.drop(['id'], axis = 1)
tweet_df 

# visualizing if there are missing values
sns.heatmap(tweet_df.isnull(), yticklabels=False, cbar=False, cmap='Blues')
# histogram for plotting positive/negative tweets (by 'label')
tweet_df.hist(bins=30, figsize=(13,5), color='r')
# or via seaborn
sns.countplot(tweet_df['label'], label='Count')

# checking the length of messages
tweet_df['length'] = tweet_df['tweet'].apply(len)
tweet_df
# plotting the length
tweet_df['length'].plot(bins=100, kind='hist')
# descriptives for length
tweet_df.describe()

# finding the shortest message
tweet_df[tweet_df['length'] == 11]['tweet'].iloc[0]
# finding the longest message
tweet_df[tweet_df['length'] == 84]['tweet'].iloc[0]

# creating a new dataset for positive tweets
positive = tweet_df[tweet_df['label']==0]
positive 
# creating a new dataset for negative tweets
negative = tweet_df[tweet_df['label']==1]
negative 

# plotting the WordCloud
sentences = tweet_df['tweet'].tolist()
sentences
len(sentences)

sentence_one_string = ''.join(sentences)

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentence_one_string))

# plotting negative sentences
n_sentences = negative['tweet'].tolist()
n_sentences
len(n_sentences)

n_sentence_one_string = ''.join(n_sentences)

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(n_sentence_one_string))

# Data Cleaning - removing punctuation
# defining a function for removing punctuation
def message_cleaning(message):
    Text_punc_removed = [char for char in message if char not in string.punctuation]
    Text_punc_removed_join = ''.join(Text_punc_removed)
    Text_punc_removed_join_clean = [word for word in Text_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Text_punc_removed_join_clean

# applying cleaning function
tweets_df_clean = tweet_df['tweet'].apply(message_cleaning)
print(tweets_df_clean[5])
# showing the original version
print(tweet_df['tweet'][5])

# applying Naive Bayes classifier to predict sentiment
vectorizer = CountVectorizer(analyzer = message_cleaning)
tweets_countvectorizer = CountVectorizer(analyzer = message_cleaning, dtype = 'uint8').fit_transform(tweet_df['tweet']).toarray()
tweets_countvectorizer.shape
X = tweets_countvectorizer
X
y = tweet_df['label']
y

# 20% of the data allocated for testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict_test))

### End of analysis
#!/usr/bin/env python
# coding: utf-8


# Importing essential libraries
import numpy as np
import pandas as pd
import pickle


# Loading the dataset
messages = pd.read_csv('feedback dataset.csv')


messages.shape


messages.info()



messages.columns



messages.head()


# # Data Cleaning & Preprocessing


messages.sentiment.unique()


def to_sentiment(sentiment):
  sentiment = sentiment
  if sentiment == 0:
    return 'negative'
  else: 
    return 'positive'
messages['target'] = messages.sentiment.apply(to_sentiment)

messages['target']


import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #to find the root word

ps = PorterStemmer()
corpus = []

for i in range (0,len(messages)):
    # Cleaning special character from the feedbacks
    feedback = re.sub('[^a-zA-Z]',' ',str(messages['text'][i]))
    
    # Converting the entire feedback into lower case
    feedback = feedback.lower()
    
    # Tokenizing the feedback by words
    feedback_words = feedback.split()
    
    # Stemming the words and removing the stopwords
    feedback = [ps.stem(word) for word in feedback_words if not word in set(stopwords.words('english')) ]
    
    # Joining the stemmed words
    feedback = ' '.join(feedback)

    # Creating a corpus
    corpus.append(feedback)

corpus[0:10]

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = messages['target'].values


X.shape


y.shape


# # Training Model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Support Vector Machine Classifier to the Training set
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)


# Predicting the Test set results
y_pred = model.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


model = SVC(C=10, gamma=0.1)
model.fit(X_train, y_train)


# # Predictions

def predict_sentiment(sample_feedback):
  sample_feedback = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_feedback)
  sample_feedback = sample_feedback.lower()
  sample_feedback_words = sample_feedback.split()
  sample_feedback_words = [word for word in sample_feedback_words if not word in set(stopwords.words('english'))]
  ps = PorterStemmer()
  final_feedback = [ps.stem(word) for word in sample_feedback_words]
  final_feedback = ' '.join(final_feedback)
  print(final_feedback)

  temp = cv.transform([final_feedback]).toarray()
  print(model.predict(temp))
  return model.predict(temp)

# Predicting values
sample_feedback = 'online learning is really worst it made me anxious.'
predict_sentiment(sample_feedback)

# Predicting values
sample_feedback = 'I find online learning best suited for me.'
predict_sentiment(sample_feedback)

pickle.dump(model,open('model.pkl','wb'))




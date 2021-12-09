#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing essential libraries
import numpy as np
import pandas as pd


# In[2]:


# Loading the dataset
messages = pd.read_csv('review.tsv', sep='\t',quoting=3)


# In[3]:


messages.shape


# In[4]:


messages.info()


# In[5]:


messages.columns


# In[6]:


messages.head()


# # Data Cleaning & Preprocessing

# In[7]:


messages.score.unique()


# In[35]:


def to_sentiment(rating):
  rating = rating
  if rating <= 2:
    return 'negative'
  elif rating == 3:
    return 'neutral'
  else: 
    return 'positive'

messages['sentiment'] = messages.score.apply(to_sentiment)


# In[36]:


import re
import nltk
nltk.download('stopwords')


# In[37]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #to find the root word


# In[38]:


ps = PorterStemmer()
corpus = []


# In[39]:


for i in range (0,len(messages)):
    # Cleaning special character from the reviews
    review = re.sub('[^a-zA-Z]',' ',str(messages['content'][i]))
    
    # Converting the entire review into lower case
    review = review.lower()
    
    # Tokenizing the review by words
    review_words = review.split()
    
    # Stemming the words and removing the stopwords
    review = [ps.stem(word) for word in review_words if not word in set(stopwords.words('english')) ]
    
    # Joining the stemmed words
    review = ' '.join(review)

    # Creating a corpus
    corpus.append(review)


# In[40]:


corpus[0:10]


# In[41]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = messages['sentiment'].values


# In[45]:


X.shape


# In[46]:


y.shape


# # Training Model

# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[48]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[49]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[55]:


# Accuracy, Precision and Recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
score1 = accuracy_score(y_test,y_pred)
print("---- Scores ----")
print("Accuracy score is: {}%".format(round(score1*100,2)))


# In[56]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[57]:


# Plotting the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize = (10,6))
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted values')
plt.ylabel('Actual values')


# In[58]:


# Hyperparameter tuning the Naive Bayes Classifier
best_accuracy = 0.0
alpha_val = 0.0
for i in np.arange(0.1,1.1,0.1):
  temp_classifier = MultinomialNB(alpha=i)
  temp_classifier.fit(X_train, y_train)
  temp_y_pred = temp_classifier.predict(X_test)
  score = accuracy_score(y_test, temp_y_pred)
  print("Accuracy score for alpha={} is: {}%".format(round(i,1), round(score*100,2)))
  if score>best_accuracy:
    best_accuracy = score
    alpha_val = i
print('--------------------------------------------')
print('The best accuracy is {}% with alpha value as {}'.format(round(best_accuracy*100, 2), round(alpha_val,1)))


# In[59]:


classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)


# # Predictions

# In[60]:


def predict_sentiment(sample_review):
  sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
  sample_review = sample_review.lower()
  sample_review_words = sample_review.split()
  sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
  ps = PorterStemmer()
  final_review = [ps.stem(word) for word in sample_review_words]
  final_review = ' '.join(final_review)
  print(final_review)

  temp = cv.transform([final_review]).toarray()
  print(classifier.predict(temp))
  return classifier.predict(temp)


# In[63]:


# Predicting values
sample_review = 'The food is really worst here.'
predict_sentiment(sample_review)


# In[64]:


# Predicting values
sample_review = 'The food is really great here.'
predict_sentiment(sample_review)


# In[ ]:





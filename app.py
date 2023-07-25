 # Importing essential libraries
import numpy as np
import pandas as pd
import pickle


# Loading the dataset
messages = pd.read_csv('feedback dataset.csv')

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

stopwords = ['and','the']

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
    feedback = [ps.stem(word) for word in feedback_words if not word in set(stopwords) ]
    #  Joining the stemmed words
    feedback = ' '.join(feedback)
    # Creating a corpus
    corpus.append(feedback)
    corpus[0:10]

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = messages['target'].values
    
# # Training Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Fitting Support Vector Machine Random Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_features='log2', n_estimators=1000)
model.fit(X_train, y_train)

# # Predictions
class Predict :
    def predict_sentiment_neg(abc,sample_feedback):
        sample_feedback = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_feedback)
        sample_feedback = sample_feedback.lower()
        sample_feedback_words = sample_feedback.split()
        sample_feedback_words = [word for word in sample_feedback_words if not word in set(stopwords)]
        ps = PorterStemmer()
        final_feedback = [ps.stem(word) for word in sample_feedback_words]
        final_feedback = ' '.join(final_feedback)
        print(final_feedback)
        temp = cv.transform([final_feedback]).toarray()
        negation = False
        if any(neg in final_feedback for neg in ["not", "n't", "no"]):
            negation =  negation
        result = model.predict(temp)
        if negation == True:
            if result == 'positive':
                result ='negative'
            elif result == 'negative':
                result = 'positive'
            else: return result
        print (result)
        return result
obj=Predict()
pickle.dump(obj,open('model.pkl','wb'))

from flask_cors import CORS, cross_origin
from flask import Flask, render_template, request, jsonify
import requests
import numpy as np
import urllib
from urllib.parse import urlencode
import json
import validators
import pickle

def _get_profane_prob(prob):
    return prob[1]


application = Flask(__name__)  # initializing a flask app
app = application

#model=pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

#route to show result 
@app.route('/report', methods=['POST', 'GET'])
@cross_origin()
def result():
     return render_template("report.html")

# route to show the predictions in a web UI
@app.route('/prediction', methods=['POST', 'GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        # try:
        te = []
        #  reading the inputs given by the user
        name=(request.form['name'])
        education=(request.form['education'])
        feedback=(request.form['feedback'])
        message=(request.form['message'])
        
        #!/usr/bin/env python
        # coding: utf-8
        object=pickle.load(open('model.pkl','rb'))
        prediction = object.predict_sentiment_neg(message)
        print(prediction)

        
        return render_template('prediction.html', prediction=prediction , name=name, message=message)
    else:
       return render_template('index.html')


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)  # running the app


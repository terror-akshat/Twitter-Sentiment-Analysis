import os
from dotenv import load_dotenv
import tweepy
from tweepy.api import API
from flask import Flask, render_template, request, jsonify, app
import pickle



load_dotenv()

# Retrieve the keys from the environment
API_KEY = os.getenv('API_KEY')
API_SECRET_KEY = os.getenv('API_SECRET_KEY')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET')

# Authenticate to Twitter
auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


app=Flask(__name__)

#Load the model 
regmodel = pickle.load(open('regression.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analysis', methods=['POST'])
def analysis():
    tweet_query = request.form['query']
    tweets = api.search_tweets(q=tweet_query, count=10)
    
    tweets_data=[]
    for tweet in tweets:
        sentiment = analysis_sentiment(tweet.text)
        tweets_data.append({'tweet':tweet.text, 'sentiment': sentiment})
    return jsonify(tweets_data)


def analysis_sentiment(tweet):
    new_data = vectorizer.transform(tweet)
    output=regmodel.predict(new_data)
    if output[0] == 1 :
        return jsonify('Positive')
    else:
        return jsonify('Negative')
    
if __name__ == '__main__':
    app.run(debug=True)
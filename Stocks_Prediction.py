import streamlit as st
import os
import http.client, urllib.parse
import pandas as pd
import pickle
import requests
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import json

@st.cache_data
def read_model():
    modelname = "ets_model.pkl"
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    loaded_model = pickle.load(open(parent_dir + "/model/" + modelname, 'rb'))
    return loaded_model

# Define the prediction function
@st.cache_data
def make_predictions(features):
    model = read_model()
    predictions = model.predict(features)
    return predictions

# Define the API call function
@st.cache_data
def fetch_news():
    # conn = http.client.HTTPConnection('api.mediastack.com')
    # params = urllib.parse.urlencode({
    #     'access_key': '65c5e9342fc45dc08165bf28b32d8ccf',
    #     'categories': 'business',
    #     'sort': 'published_desc',
    #     'limit': 10,
    #     "language": "en",
    #     "source": "CNN",
    #     "country": "gb",
    # })
    # conn.request('GET', '/v1/news?{}'.format(params))
    # res = conn.getresponse()
    # data = res.read()
    # news_data = data.decode('utf-8')
    # Load the JSON file
    parent_dir = os.path.dirname(os.path.abspath(__file__))

    with open(parent_dir + "/data/" + 'news_data.json', 'r', encoding='utf-8') as json_file:
        news_data = json.load(json_file)
    new_df = pd.json_normalize(news_data['data'])
    return new_df

# Function to process the news data and extract features
@st.cache_data
def process_news_data(news_json):
    news_df = pd.json_normalize(news_json['data'])
    
    # Example: Extract title and description to create a text feature
    news_df['text'] = news_df['title'] + " " + news_df['description']
    
    # Example: Use TF-IDF to convert text data to features
    vectorizer = TfidfVectorizer(max_features=100)
    text_features = vectorizer.fit_transform(news_df['text']).toarray()
    
    return text_features, news_df


# Streamlit App
st.image('images/universite-westminster.jpg', caption='University Of Westminster - Pragya', use_column_width=True,  channels="RGB", output_format="auto")
st.title('Pragya MSC Project - Stock Prediction using News')

st.write("""
## Fetching Latest Business News
""")

# Button to fetch news
if st.button('Fetch News'):
    news_df = fetch_news()
    st.dataframe(news_df)
    # news_json = eval(news_json)  # Convert string JSON to dictionary
    # features, news_df = process_news_data(news_df)
    
    # Predict
    # predictions = make_predictions(features)
    
    # # Add predictions to the dataframe
    # news_df['predictions'] = predictions
    
    # st.write("### News Data with Predictions")
    # st.write(news_df[['title', 'predictions']])
    
    # # Plotting the results
    # st.write("### Prediction Distribution")
    # st.bar_chart(news_df['predictions'].value_counts())

# Add footer
st.write("MSc Project")

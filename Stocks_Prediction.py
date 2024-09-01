import streamlit as st
import os
import http.client, urllib.parse
import pandas as pd
import pickle
import requests
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import matplotlib.pyplot as plt

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
    # Get today's date in YYYY-MM-DD format
    today = datetime.today().strftime('%Y-%m-%d')
    file_name = f'news_data_{today}.json'

    # Define the parent directory and file path
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(parent_dir, "data", file_name)

    # Check if the file for today's date exists
    if os.path.exists(file_path):
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as json_file:
            news_data = json.load(json_file)
        print(f"Loaded data from {file_name}")
    else:
        # Fetch news data from the API
        conn = http.client.HTTPConnection('api.mediastack.com')
        params = urllib.parse.urlencode({
            'access_key': '65c5e9342fc45dc08165bf28b32d8ccf',
            'categories': 'business',
            'sort': 'published_desc',
            "language": "en",
            "country": "gb"
        })
        conn.request('GET', '/v1/news?{}'.format(params))
        res = conn.getresponse()
        data = res.read()
        news_data = json.loads(data.decode('utf-8'))

        # Save the fetched data to a JSON file
        os.makedirs(os.path.join(parent_dir, "data"), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(news_data, json_file, ensure_ascii=False, indent=4)
        print(f"Fetched and saved data to {file_name}")

    # Convert the JSON data to a DataFrame
    news_df = pd.json_normalize(news_data['data'])
    
    # Optional: Convert 'published_at' to date if needed
    news_df['date'] = pd.to_datetime(news_df['published_at']).dt.date
    
    return news_df

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
st.image('images/FTSE.jpg', caption='University Of Westminster - Pragya', use_column_width=True,  channels="RGB", output_format="auto")

st.title('Pragya MSC Project - Stock Prediction using News')

@st.cache_data
def stock_csv():
    file_name = 'FTSE 100 Historical Results Price Data.csv'

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(parent_dir, "data", file_name)
    df = pd.read_csv(file_path)
    return df

stocks_df = stock_csv()
# Plotting the data
st.subheader(f"Stock Prices from 2012 to 2024")

# Calculate a reasonable width for the figure
fig_width = 18 

plt.figure(figsize=(20, 10))  # Adjust the width dynamically
plt.plot(stocks_df['Date'], stocks_df['Price'], color='b', marker='o', linestyle='-', label='Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Prices Over Time')
st.pyplot(plt,use_container_width=True)


st.write("""
## Fetching Latest Business News
""")
st.title('Enter the Number of news you want to fetch - ')
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

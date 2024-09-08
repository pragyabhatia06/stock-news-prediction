import streamlit as st
import os
import http.client, urllib.parse
import pandas as pd
import pickle
import numpy as np
import requests
from datetime import datetime
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from nltk.stem import WordNetLemmatizer
import os
import http.client
import urllib.parse
import json
import pandas as pd
from datetime import datetime
import re
import string
import contractions
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import nltk
 
from sklearn.preprocessing import StandardScaler
import spacy
import shutil

# Define custom NLTK data directory inside your project directory
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")

# Always delete the nltk_data directory if it exists before recreating it
if os.path.exists(nltk_data_dir):
    shutil.rmtree(nltk_data_dir)  # Remove the directory and all its contents

# Recreate the nltk_data directory
os.makedirs(nltk_data_dir, exist_ok=True)

# Add the custom NLTK data path to nltk.data.path
nltk.data.path.append(nltk_data_dir)

# Function to download NLTK resources
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
    nltk.download('vader_lexicon', download_dir=nltk_data_dir)
        
# Ensure resources are downloaded
download_nltk_data()

# Load stopwords from the custom path
stop_words = set(stopwords.words('english'))

# nlp = spacy.load("en_core_web_sm")
from spacy.cli import download

# Try to load the model and download if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Model 'en_core_web_sm' not found. Downloading it now...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

@st.cache_data
# Function to clean text
def clean_text(text):
    phrases_to_remove = [
        'new york city', 
        'new york times', 
        'rise morning',
        'huffpost rise morning newsbrief', 
        'morning newsbrief short wrapup',
        'welcome to the huffPost rise Morning Newsbrief',
        'a short wrap-up of the news to help you start your day',
        'welcome huffpost rise morning newsbrief short wrapup help start day',
        'HuffPost Rise What You Need To Know On',
    ]

    if not isinstance(text, str):
        return ""

    # Contractions expansion
    try:
        text = contractions.fix(text)
    except:
        print(text)
    
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Remove numbers
    text = re.sub(r'\d+', '', text)
    # 5.0 Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('[\\?|$|.|!-/,]', "", text)
    
    # 5.1. Remove specific phrases
    for phrase in phrases_to_remove:
        text = re.sub(re.escape(phrase.lower()), '', text)

    # 6. Tokenize text
    tokens = word_tokenize(text)

    # 7. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # 8. Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 9. Remove special characters 
    tokens = [re.sub(r'\W+', '', word) for word in tokens]
    
    # Remove single alphabet characters
    tokens = [word for word in tokens if len(word) > 1 and word.isalpha()]

    # 10. Remove extra whitespace and join tokens back into a single string
    cleaned_text = ' '.join(tokens).strip()

    return cleaned_text

@st.cache_data
def read_model():
    modelname = "VotingRegressor_final.pkl"
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    path = parent_dir + "/model/" + modelname
    try:
        loaded_model = joblib.load(path)
    except:
        # Fall back to pickle if joblib fails
        with open(path, 'rb') as file:
            loaded_model = pickle.load(file)
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
    stop_words = set(stopwords.words('english'))

    add_words = {'news','read','story','world','newsbrief','huffpost','short',
            'wrapup'}
    stop_words = stop_words.union(add_words)
    
    # Get today's date in YYYY-MM-DD format
    today = datetime.today().strftime('%Y-%m-%d')
    file_name = f'news_data_{today}.json'
    # st.write(file_name)
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
    news_df = news_df[news_df['language'] =='en']
    news_df.rename(columns = {'title':'headline','description':'short_description'},inplace=True)
    news_df = news_df[['date','headline','category','short_description','source']]

    return news_df

# Preprocessing Function
@st.cache_data
def preprocess_news(news):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenization
    tokens = word_tokenize(news.lower())
    
    # Removing stopwords and lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    
    return tokens
    

# Function to process the news data and extract features
@st.cache_data
def process_news_data(news_df):
    phrases_to_remove = [
        'new york city', 
        'new york times', 
        'rise morning',
        'huffpost rise morning newsbrief', 
        'morning newsbrief short wrapup',
        'welcome to the huffPost rise Morning Newsbrief',
        'a short wrap-up of the news to help you start your day',
        'welcome huffpost rise morning newsbrief short wrapup help start day',
        'HuffPost Rise What You Need To Know On',
    ]

    news_source_words = news_df['source'].unique()
    phrases_to_remove.extend(news_source_words)
  
    news_df['combined'] = news_df['headline'] + news_df['short_description']
    news_df['combined'] = news_df['combined'].fillna('').astype(str)
    combined_df = news_df.groupby('date').agg({
        'combined': ' '.join
    }).reset_index()
    combined_df['clean_title'] = combined_df['combined'].apply(clean_text)    

    comment_combined =''
    for each in combined_df['clean_title']:
        comment_combined = comment_combined + " " + each

    startofdoc = 0
    endofdoc = 1000000

    df_ner = pd.DataFrame(columns=['Text','Label'])
    st.title('Name Entity Recognition')

    while startofdoc < len(comment_combined) :
        doc = nlp(comment_combined[startofdoc:endofdoc])

        for ent in doc.ents:
            df_ner.loc[len(df_ner)]  = [ent.text, ent.label_]
        startofdoc = startofdoc + 1000000 + 1
        endofdoc = endofdoc + 1000000 - 2
    
    for name_val in df_ner['Label'].unique():
        try:
            xdf = pd.DataFrame(df_ner[df_ner['Label'] ==name_val]['Text'].value_counts()[:10]).reset_index()
            xdf = xdf.rename(columns = {'index':name_val})
            xdf = xdf.rename(columns = {'Text':'frequency'})

            # Debug print statements
            st.write(name_val + " Entity Analysis")
            st.dataframe(xdf)
        except: pass
    
    df_topic = combined_df.copy()

    # Apply preprocessing
    df_topic['tokens'] = df_topic['clean_title'].apply(preprocess_news)
    df_topic

    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(df_topic['tokens'])

    # Create a corpus: List of lists of (token_id, token_count)
    corpus = [dictionary.doc2bow(text) for text in df_topic['tokens']]

    optimal_num_topics = 16

        # Train the LDA model
    lda_model = LdaModel(corpus=corpus, 
                        id2word=dictionary, 
                        num_topics=optimal_num_topics, 
                        random_state=42, 
                        passes=10,
                        update_every=1,
                        alpha='auto',
                        per_word_topics=True
                        )
    
    st.title('Topic Creation')
    # Get the topics from the best LDA model
    topics = lda_model.print_topics(-1)

    # Function to clean up the topics
    def clean_topic(topic):
        # Remove the numeric weights (e.g., '0.007*') using a regular expression
        cleaned_topic = ''.join([word.split("*")[1] for word in topic.split(" + ")])
        cleaned_topic = cleaned_topic.replace('"',' ')
        return cleaned_topic.strip(' ')  # Remove any extra quotation marks

    for topic in topics:
        st.write(f"Topic {topic[0]}: {clean_topic(topic[1])}")
    
    # Assign topic distribution to each day

    @st.cache_data
    def get_topic_distribution(text):
        bow = dictionary.doc2bow(text)
        return lda_model.get_document_topics(bow)

    df_topic['topic_distribution'] = df_topic['tokens'].apply(get_topic_distribution)

    @st.cache_data
    def aggregate_topics_by_date(group):
        topic_sums = defaultdict(float)
        for dist in group['topic_distribution']:
            for topic_id, prob in dist:
                topic_sums[topic_id] += prob
        total = sum(topic_sums.values())
        return {f"topic_{k}": v/total for k, v in topic_sums.items()}

    # Group by date and aggregate topics
    topic_features = df_topic.groupby('date').apply(aggregate_topics_by_date).apply(pd.Series).reset_index()

    df_ner_columns = [ 'NORP', 'PERSON', 'ORG','CARDINAL','GPE',
        'ORDINAL','TIME', 'LOC', 'EVENT', 'LAW',  'FAC',
                  'MONEY','PRODUCT', 'QUANTITY', 'PERCENT']
    top_ner = 5
    df_ner = df_ner[df_ner['Label'].isin(df_ner_columns)]
    df_ner = df_ner.rename(columns= {'Text':'entity','Label':'label'})

    # Group by 'label' and 'entity' to count occurrences
    entity_counts = df_ner.groupby(['label', 'entity']).size().reset_index(name='count')

    # For each label, get the top 3 entities based on frequency
    top_entities = entity_counts.groupby('label').apply(lambda x: x.nlargest(top_ner, 'count')).reset_index(drop=True)

    # Convert to a dictionary for easy lookup
    top_entities_dict = top_entities.groupby('label')['entity'].apply(list).to_dict()
    df_ner_feat = combined_df.copy()

    # Initialize columns for the top NER entities
    for label, entities in top_entities_dict.items():
        for entity in entities:
            df_ner_feat[f'{label}_{entity}'] = df_ner_feat['clean_title'].apply(lambda x: 1 if entity in x else 0)

    df_clust = combined_df.copy()

    # Function to extract entities and key phrases
    @st.cache_data
    def extract_phrases(text):
        doc = nlp(text)
        # Extract named entities and noun chunks (key phrases)
        entities = [ent.text for ent in doc.ents]
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        return entities + noun_chunks

    # Apply the function to the news articles
    df_clust['phrases'] = df_clust['clean_title'].apply(extract_phrases)

    # Flatten the list of phrases for vectorization
    df_clust['phrases_flat'] = df_clust['phrases'].apply(lambda x: ' '.join(x))

    # Vectorize the phrases
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_clust['phrases_flat'])

    best_eps = 200
    best_min_samples = 2

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    df_clust['event_cluster'] = dbscan.fit_predict(X.toarray())

    # Label events as significant if they belong to a cluster
    df_clust['is_event'] = df_clust['event_cluster'].apply(lambda x: 1 if x != -1 else 0)

    sia = SentimentIntensityAnalyzer()
    @st.cache_data
    def score_vader(text):
        return sia.polarity_scores(text)["compound"]
    combined_df['vader_sentiment'] = combined_df['clean_title'].apply(score_vader)

    @st.cache_data
    def score_textblob(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity  # You can also include subjectivity if needed

    combined_df['textblob_sentiment'] = combined_df['clean_title'].apply(score_textblob)
    df_clean = combined_df.copy()
    df_clean = df_clean[['date','vader_sentiment','textblob_sentiment']]
    df_clean = df_clean.merge(df_clust[['date','is_event']], on ='date')
    df_clean = df_clean.merge(topic_features, on ='date')
    df_clean = df_clean.merge(df_ner_feat, on ='date')
    df_clean.drop(columns=['combined','clean_title'],inplace=True)
    df_clean.fillna(0,inplace=True)
    return df_clean
    


# Streamlit App
st.image('images/universite-westminster.jpg', caption='University Of Westminster - Pragya', use_column_width=True,  channels="RGB", output_format="auto")
st.image('images/FTSE.jpg', caption='University Of Westminster - Pragya', use_column_width=True,  channels="RGB", output_format="auto")
st.write('-----------------------------------')
st.title('Pragya MSC Project - Stock Prediction using News')
st.write('-----------------------------------')
st.title('Supervisor - Tamas Kiss')
st.write('-----------------------------------')

@st.cache_data
def stock_csv():
    file_name = 'FTSE 100 Historical Results Price Data.csv'

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(parent_dir, "data", file_name)
    df = pd.read_csv(file_path)
   # df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is datetime format --changed
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df['time_index'] = df['Date']
    # Convert 'Date' to index
    df.set_index('time_index', inplace=True)
    df = df.sort_values('Date')
    return df

@st.cache_data
def historical_data_csv():
    file_name = 'historical_data.csv'

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(parent_dir, "data", file_name)
    df = pd.read_csv(file_path)
    df['time_index'] = pd.to_datetime(df['time_index'])
    df['date'] = pd.to_datetime(df['time_index'])  # Ensure 'Date' is datetime format
    # Convert 'Date' to index
    df.set_index('time_index', inplace=True)
    df = df.sort_values('date')

    # Step 2: Now, you can format the 'date' column to 'dd/mm/yyyy'
    df['date'] = df['date'].dt.strftime('%d/%m/%Y')
    df = df.sort_values(by='date')
    return df

stocks_df = stock_csv()
# Plotting the data
# Assuming stocks_df is your DataFrame and it includes 'Date' and 'Price' columns
st.subheader(f"Stock Prices from 2012 to 2024")

# Prepare the data for st.line_chart

stocks_df = stocks_df.sort_index(ascending=True)

# Streamlit app
st.title("Stock Price Over Time")
# Convert columns to numeric
# Remove commas from the columns and convert to numeric
stocks_df[['Price', 'Open', 'High', 'Low']] = stocks_df[['Price', 'Open', 'High', 'Low']].replace({',': ''}, regex=True)

# Now convert to numeric
stocks_df[['Price', 'Open', 'High', 'Low']] = stocks_df[['Price', 'Open', 'High', 'Low']].apply(pd.to_numeric, errors='coerce')


# Display the line chart using st.line_chart
# Use st.line_chart to create a line chart
st.line_chart(stocks_df[['Price', 'Open', 'High', 'Low']])



df = historical_data_csv()
# # st.dataframe(df)
st.title('Forecast Duration')
forecast_duration = st.number_input('Enter the Forecast Duration',max_value=365,value=7)

future_dates = pd.date_range(start=pd.Timestamp.today(), periods=forecast_duration, freq='D')
future_dates = future_dates.strftime('%m/%d/%Y')
future_df = pd.DataFrame({'date': future_dates})
for col in df.columns:
    if col != 'date':
        future_df[col] = np.nan
df_combined = pd.concat([df, future_df], ignore_index=True)

@st.cache_data
# Function to fill NaN values using the mean of the previous `forecast_duration` values
def fill_with_previous_forecast_duration(series, forecast_duration=7):
    filled_series = series.copy()
    for i in range(len(series)):
        if pd.isna(series.iloc[i]):  # Check if the current value is NaN
            # Get previous `forecast_duration` values
            previous_values = series[max(0, i-forecast_duration):i]
            if len(previous_values) > 0:  # Ensure there are values to calculate the mean
                # Calculate the mean of the previous values, ignoring NaN values
                mean_value = previous_values.mean()
                filled_series.iloc[i] = mean_value
    return filled_series

# Apply the function to each column in the DataFrame
for col in df_combined.columns:
    # print(f"Processing column: {col}")
    df_combined[col] = fill_with_previous_forecast_duration(df_combined[col])


# Forward fill the NaN values with the last known value --changed
#df_combined.fillna(method='ffill', inplace=True)
#df_combined.fillna(method='bfill', inplace=True)
###########
df_combined.ffill(inplace=True) 
df_combined.bfill(inplace=True)
#########
df_combined = df_combined.fillna(df_combined.mean(numeric_only=True))

df_combined = df_combined[-forecast_duration:]
# df_combined = df_combined[df.columns]
df_combined.set_index('date', inplace=True)
# Standardize the features manually
scaler = StandardScaler()

# st.dataframe(df_combined)
# Fit the scaler on the training data and transform the training data
X_train_scaled = scaler.fit_transform(df_combined)

# st.dataframe(X_train_scaled)

model = read_model()
# st.write(len(df_combined.columns))
prediction_result = stocks_df[-forecast_duration:]
prediction_result = model.predict(X_train_scaled)

# st.dataframe(prediction_result)

#  -----------------------------

# Assuming stock_df is your original DataFrame with time series data and df_combined is the truncated DataFrame
# st.write('test')
last_data = stocks_df[-forecast_duration:]
# st.dataframe(last_data)
# Ensure df_combined and prediction_result are aligned in terms of index
# If df_combined is already aligned, we create a new DataFrame for the predictions
prediction_df = pd.DataFrame(prediction_result, index=df_combined.index, columns=['Predicted'])
# st.dataframe(prediction_df)
# Combine the original last data and the prediction
combined_df = pd.concat([last_data, prediction_df], axis=1)

# Step 1: Convert time index to a uniform date format (e.g., YYYY-MM-DD)
combined_df.index = pd.to_datetime(combined_df.index).strftime('%Y-%m-%d')

# Step 2: Merge the "Price" and "Predicted" columns
combined_df['Price_Merged'] = combined_df['Price'].combine_first(combined_df['Predicted'])

# New Code: Ensure 'Price_Merged' is numeric to avoid Arrow serialization issues --changed
combined_df['Price_Merged'] = pd.to_numeric(combined_df['Price_Merged'], errors='coerce')

# Drop the individual "Price" and "Predicted" columns if needed
combined_df = combined_df.drop(columns=['Price', 'Predicted'])

# Fill any remaining NaN values --changed
#combined_df.fillna(method='ffill', inplace=True)
#combined_df.fillna(method='bfill', inplace=True)
####
combined_df.ffill(inplace=True)
combined_df.bfill(inplace=True)
####
# Get today's date
today_date = datetime.today().strftime('%Y-%m-%d')

# Print the heading with today's date
st.title(f"Stock Prediction - {today_date}")
st.line_chart(combined_df['Price_Merged'])
# ------------------------------------------

import requests
from bs4 import BeautifulSoup

@st.cache_data
def fetch_ftse_100_price():
    # URL of the website you want to scrape
    url = "https://finance.yahoo.com/quote/%5EFTSE/"

    # Send a GET request to the website
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate the element that contains the FTSE 100 price
        # The exact selector will depend on the website's structure
        price_tag = soup.find('div', class_='price')  # This is an example, you need to inspect the page to find the correct class or tag

        if price_tag:
            price = price_tag.text.strip()
            st.title(f"FTSE 100 Index Price: {price}")
            url = 'https://finance.yahoo.com/quote/%5EFTSE/'
            st.write("Yahoo Finance FTSE 100 - [link](%s)" % url)
            
            return price
        else:
            print("Could not find the FTSE 100 price on the page.")
            return None
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None

# Example usage
fetch_ftse_100_price()

st.title('Latest News ')
news_df1 = fetch_news()
st.dataframe(news_df1)

if st.button('Update the model with new features'):
    news_df_Predict = process_news_data(news_df1)
    st.title('Features Found')
    st.dataframe(news_df_Predict)

# Add footer
st.write("MSc Project By - Pragya Bhatia")

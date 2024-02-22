import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import download
from nltk.corpus import wordnet

# Check if NLTK data is present, if not, download it
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    download('stopwords')

try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    download('punkt')

# Set up NLTK resources
stopwords.words('english')
wordnet.ensure_loaded()

# Load the logistic regression model and TF-IDF vectorizer
lr_model = joblib.load('logistic_regression_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Define a lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the predict_sentiment function
def predict_sentiment(input_text):
    preprocessed_text = perform_lemmatization(input_text)
    vectorized_input = tfidf_vectorizer.transform([preprocessed_text])
    prediction = lr_model.predict(vectorized_input)
    if prediction[0] == 'Positive':
        return "Positive"
    else:
        return "Negative"

# Define the lemmatization function
def perform_lemmatization(content):
    words = word_tokenize(re.sub('[^a-zA-Z]', ' ', content.lower()))
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

# Set the title of the app
st.title('Sentiment Analysis')

# Create a text input field for user input
input_text = st.text_area('Enter your text:', '')

# Add a button to trigger sentiment prediction
if st.button('Predict'):
    # Perform sentiment prediction when the button is clicked
    predicted_sentiment = predict_sentiment(input_text)
    st.write('Predicted sentiment:', predicted_sentiment)

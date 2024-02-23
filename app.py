import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import download
from nltk.corpus import wordnet


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
lemmatizer = WordNetLemmatizer()

def predict_sentiment(input_text):
    preprocessed_text = perform_lemmatization(input_text)
    vectorized_input = tfidf_vectorizer.transform([preprocessed_text])
    prediction = lr_model.predict(vectorized_input)
    if prediction[0] == 'Positive':
        return "Positive"
    else:
        return "Negative"

def perform_lemmatization(content):
    words = word_tokenize(re.sub('[^a-zA-Z]', ' ', content.lower()))
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text

# Set the title of the app
st.title('Sentiment Analysis')

# Add a note as placeholder text inside the input field
placeholder_text = "This model is in its rudimentary stage. Can detect only 'Positive' or 'Negative' sentiments(..yet). It has not been trained on neutral statements and will not give accurate results if entered.\n\nFor starters, type \"I really loved that movie! it was amazing\" and hit the button!"
input_text = st.text_area('Enter your text:', value='', placeholder=placeholder_text)

# Add a button to trigger sentiment prediction
if st.button('Predict'):
    # Perform sentiment prediction when the button is clicked
    predicted_sentiment = predict_sentiment(input_text)
    st.write('Predicted sentiment:', predicted_sentiment)

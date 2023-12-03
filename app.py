import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

model = load_model('GRU.h5') # Loading the pretrained GRU model

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data) # Loading the tokenizer

MAX_SEQUENCE_LENGTH = 20

def preprocess_text(text):
    # Convert the text to a sequence
    sequence = tokenizer.texts_to_sequences([text])
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded_sequence

def predict_sentiment(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    # Make a prediction
    prediction = model.predict(processed_text)
    sentiment = ['Positive', 'Negative', 'Neutral']
    return sentiment[np.argmax(prediction)]

st.title('Sentiment Analysis App')

user_input = st.text_area("Enter the text you'd like to analyze for sentiment:")

if st.button('Analyze'):
    result = predict_sentiment(user_input)
    st.write(f'The predicted sentiment is: {result}')

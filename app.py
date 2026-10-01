import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import sys

# Add src folder to path to import our cleaning function
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data_processing import clean_text

# Paths
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎥")

st.title("🎥 Movie Review Sentiment Analysis")
st.write("Built with a deep learning CNN + LSTM model! Enter a review below to see if it's Positive or Negative.")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

if model is None or tokenizer is None:
    st.warning("⚠️ Model or tokenizer not found. Please make sure to train the model first by running `python src/train.py`")
else:
    user_input = st.text_area("Write your review here:", "This movie was absolutely amazing! I loved every second of it.")
    
    if st.button("Predict Sentiment"):
        with st.spinner("Analyzing..."):
            if user_input.strip() == "":
                st.error("Please enter a valid review.")
            else:
                # Preprocess the input text (Step 4, 5, 6 equivalent)
                cleaned = clean_text(user_input)
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=200, padding='post')
                
                # Predict
                prediction = model.predict(padded)[0][0]
                
                # Display output
                if prediction >= 0.5:
                    st.success(f"🎉 **Positive Review!** (Confidence: {prediction:.2%})")
                else:
                    st.error(f"😞 **Negative Review!** (Confidence: {(1-prediction):.2%})")

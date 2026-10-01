import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def load_and_explore_data(filepath):
    """Step 2 & 3: Load and explore the dataset."""
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head())
    print("\nSentiment Distribution:\n", df['sentiment'].value_counts())
    
    # Convert sentiments to binary (Assuming standard Kaggle IMDB dataset format)
    df['sentiment'] = (df['sentiment'].astype(str).str.lower() == 'positive').astype(int)
    return df

def clean_text(text):
    """Step 4: Clean/preprocess text."""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

def preprocess_dataframe(df):
    """Applies text cleaning to the dataframe."""
    print("Cleaning text data... this might take a minute.")
    df['clean_review'] = df['review'].apply(clean_text)
    return df

def tokenize_and_pad(df, max_words=10000, max_len=200):
    """Step 5 & 6: Tokenization and Padding."""
    print("Tokenizing and padding sequences...")
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['clean_review'])
    
    # Step 5: Tokenization
    sequences = tokenizer.texts_to_sequences(df['clean_review'])
    
    # Step 6: Padding
    X = pad_sequences(sequences, maxlen=max_len, padding='post')
    y = np.array(df['sentiment'].tolist(), dtype=int)
    
    return X, y, tokenizer

def save_tokenizer(tokenizer, filepath="tokenizer.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {filepath}")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "IMDB Dataset.csv")
    
    if not os.path.exists(filepath):
         print(f"Error: Could not find '{filepath}'.")
         print("Please download the IMDB Dataset CSV (e.g. from Kaggle) and place it in the project root folder as 'IMDB Dataset.csv'.")
    else:
        df = load_and_explore_data(filepath)
        df = preprocess_dataframe(df)
        X, y, tokenizer = tokenize_and_pad(df)
        
        # Save tokenizer for later use in our simple application
        save_tokenizer(tokenizer, os.path.join(base_dir, "tokenizer.pkl"))
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Data preparation complete!")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

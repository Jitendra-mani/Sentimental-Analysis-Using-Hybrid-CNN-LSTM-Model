import os
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_processing import load_and_explore_data, preprocess_dataframe, tokenize_and_pad, save_tokenizer, clean_text
from model import build_cnn_lstm_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "IMDB Dataset.csv")
    model_save_path = os.path.join(base_dir, "sentiment_model.h5")
    tokenizer_path = os.path.join(base_dir, "tokenizer.pkl")
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please download it.")
        return

    # Data Preparation
    print("--- Preparing Data ---")
    df = load_and_explore_data(data_path)
    df = preprocess_dataframe(df)
    
    vocab_size = 10000
    max_len = 200
    X, y, tokenizer = tokenize_and_pad(df, max_words=vocab_size, max_len=max_len)
    save_tokenizer(tokenizer, tokenizer_path)
    
    # Split Data (Train, Val, Test)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)
    
    # Build Model
    print("\n--- Building Model ---")
    model = build_cnn_lstm_model(vocab_size=vocab_size, max_len=max_len)
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)
    
    # Step 13 & 14: Train and Validate model
    print("\n--- Training Model ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=128,
        callbacks=[early_stop, checkpoint]
    )
    
    # Step 15 & 16: Test model and Evaluate metrics
    print("\n--- Evaluating Model ---")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Step 18: Save model is handled by ModelCheckpoint callback above
    print(f"\nModel saved to {model_save_path}")
    
    # Step 17: Test custom reviews
    print("\n--- Testing Custom Reviews ---")
    custom_reviews = [
        "This movie was absolutely wonderful. The acting was great and the plot was engaging.",
        "Terrible waste of time. The script was awful and the directing was worse.",
        "This is the best movie i have ever seen.","This movie is not my cup of tea as it is boring."
    ]
    
    for review in custom_reviews:
        # Preprocess exactly like the training data
        cleaned = clean_text(review)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        
        pred_prob = model.predict(padded)[0][0]
        sentiment = "Positive" if pred_prob > 0.5 else "Negative"
        print(f"Review: '{review}'\nPrediction: {sentiment} ({pred_prob:.4f})\n")

if __name__ == "__main__":
    train_and_evaluate()

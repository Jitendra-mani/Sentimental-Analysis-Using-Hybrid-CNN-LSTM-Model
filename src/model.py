import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout

def build_cnn_lstm_model(vocab_size=10000, max_len=200, embedding_dim=128):
    """
    Builds the CNN + LSTM Sentiment Analysis Model (Steps 7 to 12).
    """
    model = Sequential()
    
    # Step 7: Word Embedding
    # Maps our vocabulary index into dense vectors of fixed size
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len))
    
    # Step 8: Build CNN
    # Convolutional layer for extracting spatial features/patterns from words
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    
    # Step 9: Add MaxPooling
    # Downsamples the feature maps
    model.add(MaxPooling1D(pool_size=4))
    
    # Step 10: Add LSTM
    # Long Short-Term Memory layer to learn sequential dependencies
    model.add(LSTM(64, return_sequences=False))
    
    # Dropout to prevent overfitting
    model.add(Dropout(0.5))
    
    # Step 11: Add output layer
    # Sigmoid activation for binary classification (positive vs negative)
    model.add(Dense(1, activation='sigmoid'))
    
    # Step 12: Compile model
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Test model building
    model = build_cnn_lstm_model()
    model.summary()
    print("Model compilation complete!")

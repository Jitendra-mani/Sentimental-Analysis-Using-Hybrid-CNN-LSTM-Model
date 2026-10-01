# IMDb Sentiment Analysis

A complete end-to-end sentiment analysis project classifying movie reviews using a Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) layers.

## Roadmap Accomplished
1. Setup Python environment (`requirements.txt`)
2. Get IMDb dataset (Requires manual download of Kaggle CSV)
3. Explore the dataset (`src/data_processing.py`)
4. Clean/preprocess text (`src/data_processing.py`)
5. Tokenization (`src/data_processing.py`)
6. Padding (`src/data_processing.py`)
7. Word Embedding (`src/model.py`)
8. Build CNN (`src/model.py`)
9. Add MaxPooling (`src/model.py`)
10. Add LSTM (`src/model.py`)
11. Add output layer (`src/model.py`)
12. Compile model (`src/model.py`)
13. Train model (`src/train.py`)
14. Validate model (`src/train.py`)
15. Test model (`src/train.py`)
16. Evaluate metrics (`src/train.py`)
17. Test custom reviews (`src/train.py`)
18. Save model (`src/train.py`)
19. Create simple application (`app.py`)
20. Prepare project explanation (`README.md`)

---

## 🚀 Getting Started

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
1. Download the **IMDB Dataset of 50k Movie Reviews** (usually a `.csv` file).
2. Rename it to `IMDB Dataset.csv`.
3. Place it in the root folder of this project (next to this README).

### 3. Train the Model
Run the training script. This script handles data loading, preprocessing, model compiling, training, evaluating, and ultimately saves the tokenizer and `sentiment_model.h5`.
```bash
python src/train.py
```

### 4. Run the Web Application
Once training is done and the `.h5` file is saved, start the Streamlit app to interact with the model:
```bash
streamlit run app.py
```
Open the provided local URL in your browser, enter a review, and get a real-time sentiment prediction!

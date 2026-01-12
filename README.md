# Text Analytics Project

This project contains two main notebooks for text analytics tasks:

## Part 1: TF-IDF Lyric Analysis (`tf-idf_lyric_analysis.ipynb`)

This notebook implements comprehensive TF-IDF analysis on song lyrics:

1. **Data Preparation**: 
   - Three English songs (Bohemian Rhapsody, Hotel California, Stairway to Heaven)
   - Text preprocessing: lowercase, stopword removal, lemmatization, tokenization

2. **TF-IDF Implementation**:
   - Calculates TF-IDF matrix using sklearn
   - Visualizes top terms per song
   - Creates histograms and box plots

3. **Comparison with Other Methods**:
   - Count Vectorizer
   - Word2Vec
   - Doc2Vec
   - Comparison based on computational complexity, representation quality, and interpretability

4. **Statistical Analysis**:
   - Top 10 most frequent words
   - Top 10 most frequent bigrams
   - WordCloud visualizations
   - t-SNE dimensionality reduction

## Part 2: BERT Sentiment Classification (`bert_sentiment_classification.ipynb`)

This notebook implements BERT-based sentiment classification on IMDB movie reviews:

1. **Data Preparation**:
   - Loads IMDB Dataset.csv
   - Text preprocessing (HTML tag removal)
   - Train/test split (80/20)

2. **BERT Model Setup**:
   - Downloads pre-trained BERT model (`bert-base-uncased`) from Hugging Face
   - Loads BERT tokenizer
   - Creates custom dataset class with proper tokenization

3. **Model Training**:
   - Fine-tunes BERT for binary sentiment classification
   - Uses AdamW optimizer with learning rate scheduling
   - Tracks training history

4. **Evaluation**:
   - Calculates accuracy, precision, recall, F1-score
   - Creates confusion matrix
   - Manual inspection of examples
   - Inference time testing
   - Model stability testing

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (will be done automatically in notebooks):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

1. **For TF-IDF Analysis**:
   - Open `tf-idf_lyric_analysis.ipynb`
   - Run all cells sequentially
   - Visualizations will be saved automatically

2. **For BERT Classification**:
   - Ensure `IMDB Dataset.csv` is in the same directory
   - Open `bert_sentiment_classification.ipynb`
   - Run all cells sequentially
   - Training may take some time depending on your hardware
   - Model will download from Hugging Face on first run

## Requirements

- Python 3.8+
- PyTorch (CPU or GPU version)
- Transformers library from Hugging Face
- See `requirements.txt` for full list

## Notes

- The BERT model will be downloaded automatically from Hugging Face (~440MB)
- Training BERT may take 30-60 minutes on CPU, much faster on GPU
- All visualizations are saved as PNG files in the project directory


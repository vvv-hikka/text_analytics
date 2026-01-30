# Text Analytics Project

This project contains two main notebooks for text analytics tasks:

## Part 1: TF-IDF Lyric Analysis (`tf-idf_lyric_analysis.ipynb`)

This notebook implements comprehensive TF-IDF analysis on song lyrics:

1. **Data Preparation**: 
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
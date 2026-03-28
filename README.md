# Fake News Detector

A machine learning project that detects whether a news article is Real or Fake using NLP and Logistic Regression.

## Problem Statement
Misinformation and fake news have become a major social problem in the digital age. This project automatically classifies news articles as real or fake using machine learning.

## Dataset
- Name: WELFake Dataset
- Source: Kaggle
- Size: 72,134 news articles
- Labels: 0 = Real, 1 = Fake

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Matplotlib and Seaborn
- Jupyter Notebook

## How to Run

Step 1 - Clone the repository
git clone https://github.com/divyanshukumar3740-cmd/fake-news-detector.git

Step 2 - Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

Step 3 - Download the dataset from Kaggle and place WELFake_Dataset.csv in the data folder
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Step 4 - Run the notebook
jupyter notebook notebooks/fake_news_detector.ipynb

## Results
- Accuracy: 94.36%
- Precision for Real News: 0.95
- Precision for Fake News: 0.94
- F1-Score: 0.94

## How It Works
1. Load and clean the dataset
2. Combine title and text into one content column
3. Convert text to numbers using TF-IDF
4. Train a Logistic Regression model
5. Evaluate using accuracy and confusion matrix
6. Test on custom news articles

## Project Structure
- data folder - contains the dataset
- notebooks folder - contains the Jupyter notebook
- model.pkl - saved trained model
- tfidf.pkl - saved TF-IDF vectorizer
- README.md - project documentation

## Author
Divyanshu Kumar
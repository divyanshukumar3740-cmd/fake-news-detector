# Fake News Detector

A machine learning project that detects whether a news article is Real or Fake using NLP and Logistic Regression.

## Problem Statement

Misinformation and fake news have become a major social problem in the digital age. This project automatically classifies news articles as real or fake using machine learning.

## Dataset

* Name: WELFake Dataset (sample)
* Source: Kaggle
* Size: 10,000 articles
* Labels: 0 = Real, 1 = Fake

## Tech Stack

* Python
* Pandas and NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Matplotlib and Seaborn
* Jupyter Notebook

## How to Run

Step 1 - Clone the repository

git clone https://github.com/divyanshukumar3740-cmd/fake-news-detector.git

cd fake-news-detector

Step 2 - Install dependencies

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Step 3 - Train the model

python main.py --train

Step 4 - Predict on a news headline

python main.py --predict "your news headline here"

Example:

python main.py --predict "Scientists discover coffee makes you immortal!"

## Results

* Accuracy: 90.9%
* Precision for Real News: 0.92
* Precision for Fake News: 0.90
* F1-Score: 0.91

## How It Works

1. Load and clean the dataset
2. Combine title and text into one content column
3. Convert text to numbers using TF-IDF
4. Train a Logistic Regression model
5. Evaluate using accuracy and confusion matrix
6. Test on custom news articles

## Project Structure

* data folder - contains the dataset
* notebooks folder - contains the Jupyter notebook
* model.pkl - saved trained model
* tfidf.pkl - saved TF-IDF vectorizer
* README.md - project documentation

## Author

Divyanshu Kumar


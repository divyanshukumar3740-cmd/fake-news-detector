import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = 'data/WELFake_Dataset.csv'
MODEL_PATH = 'model.pkl'
TFIDF_PATH = 'tfidf.pkl'

def train():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Cleaning data...")
    df = df.dropna(subset=['title', 'text'])
    df['content'] = df['title'] + ' ' + df['text']

    X = df['content']
    y = df['label']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing text using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {round(accuracy * 100, 2)}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump(tfidf, f)

    print("Model saved successfully!")

def predict(news_text):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
        print("Model not found. Please run training first using: python main.py --train")
        return

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(TFIDF_PATH, 'rb') as f:
        tfidf = pickle.load(f)

    transformed = tfidf.transform([news_text])
    prediction = model.predict(transformed)[0]
    probability = model.predict_proba(transformed)[0]
    label = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
    confidence = round(max(probability) * 100, 2)

    print(f"\nInput: {news_text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence}%")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train model:   python main.py --train")
        print("  Predict news:  python main.py --predict \"your news headline here\"")
    elif sys.argv[1] == '--train':
        train()
    elif sys.argv[1] == '--predict' and len(sys.argv) >= 3:
        predict(sys.argv[2])
    else:
        print("Invalid command. Use --train or --predict \"news text\"")
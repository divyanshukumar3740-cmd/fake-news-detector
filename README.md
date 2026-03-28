\# 📰 Fake News Detector



A machine learning project that detects whether a news article is \*\*Real or Fake\*\* using Natural Language Processing (NLP) and Logistic Regression.



\## 📌 Problem Statement

Misinformation and fake news have become a major threat in today's digital world. This project aims to automatically classify news articles as real or fake using machine learning techniques.



\## 📊 Dataset

\- \*\*Name:\*\* WELFake Dataset

\- \*\*Source:\*\* Kaggle

\- \*\*Size:\*\* 72,134 news articles

\- \*\*Labels:\*\* 0 = Real, 1 = Fake



\## 🛠️ Tech Stack

\- Python

\- Pandas, NumPy

\- Scikit-learn

\- TF-IDF Vectorizer

\- Logistic Regression

\- Matplotlib, Seaborn

\- Jupyter Notebook



\## 🚀 How to Run



\### 1. Clone the repository

```

git clone https://github.com/divyanshukumar3740-cmd/fake-news-detector.git

cd fake-news-detector

```



\### 2. Install dependencies

```

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

```



\### 3. Download the dataset

Download WELFake\_Dataset.csv from Kaggle and place it in the `data/` folder:

👉 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset



\### 4. Run the notebook

```

jupyter notebook notebooks/fake\_news\_detector.ipynb

```



\## 📈 Results

| Metric | Score |

|--------|-------|

| Accuracy | 94.36% |

| Precision (Real) | 0.95 |

| Precision (Fake) | 0.94 |

| F1-Score | 0.94 |



\## 🔍 How It Works

1\. Load and clean the dataset

2\. Combine title and text into a single content column

3\. Convert text to numerical features using TF-IDF

4\. Train a Logistic Regression classifier

5\. Evaluate using accuracy, precision, recall and confusion matrix

6\. Test on custom news articles



\## 📁 Project Structure

```

fake-news-detector/

├── data/

│   └── WELFake\_Dataset.csv

├── notebooks/

│   └── fake\_news\_detector.ipynb

├── model.pkl

├── tfidf.pkl

└── README.md

```



\## 👤 Author

Divyanshu Kumar


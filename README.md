# fake-news-detection-7
# fake-news-detection-7
# 📰 AI-Powered Fake News Detection System

[![Live Demo](https://img.shields.io/badge/Live_Demo-View_Project-blue?style=for-the-badge&logo=google-chrome)](https://sejalyadav3815.github.io/fake-news-detection-7/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/sejal-yadav-3a7a29328)

A real-time web application that detects fake news using **Machine Learning** and **Natural Language Processing**, achieving **97.35% accuracy** on a dataset of over 44,000 news articles.

![Fake News Detection Demo](https://img.shields.io/badge/Accuracy-97.35%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)

## 🚀 Live Demo

Experience the application here: **[Fake News Detection System](https://sejalyadav3815.github.io/fake-news-detection-7/)**

## ✨ Key Features

- **⚡ Real-time Analysis:** Instantly classify news as *FAKE* or *REAL*.
- **📊 Confidence Score:** Displays the model's confidence percentage for each prediction.
- **📜 Analysis History:** Stores your last 20 predictions with timestamps.
- **🎨 Modern UI:** Responsive design with smooth animations (works on mobile and desktop).
- **🧠 Advanced ML:** Uses **Naive Bayes** classifier with **TF-IDF** feature extraction.
- **🔍 NLP Processing:** Text cleaning, stopword removal, and punctuation handling.

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Python, Flask |
| **Machine Learning** | scikit-learn, NLTK, Pandas, NumPy |
| **Deployment** | GitHub Pages (Frontend), PythonAnywhere (Backend API) |
| **Version Control** | Git, GitHub |

## 📊 Model Performance

- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Dataset:** 44,000+ news articles (20,000 fake / 24,000 real)
- **Accuracy:** **97.35%**
- **Precision:** 97.2% (Fake), 97.5% (Real)
- **Recall:** 97.1% (Fake), 97.6% (Real)

## 🚀 How It Works

1. **Text Input:** User pastes a news article into the web interface.
2. **Preprocessing:** Text is cleaned, lowercased, punctuation removed, and stopwords filtered.
3. **Feature Extraction:** TF-IDF converts text into numerical features.
4. **Prediction:** The trained Naive Bayes model classifies the article.
5. **Result:** Prediction (FAKE/REAL) and confidence score are displayed.

## 📁 Project Structure



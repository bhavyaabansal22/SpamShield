# 🛡️ SpamShield

### Intelligent Email & SMS Spam Detection using Machine Learning

<p align="center">
  <b>Detect spam messages instantly using NLP, TF-IDF and Multinomial Naive Bayes.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/NLP-NLTK-4B8BBE?style=for-the-badge">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
</p>

---

## ✨ Overview

**SpamShield** is a machine learning-based web application that classifies **SMS and email messages as Spam or Not Spam**.

The application combines **Natural Language Processing (NLP)** with **TF-IDF feature extraction** and a **Multinomial Naive Bayes classifier** to analyze the content of a message and make a prediction.

> 📩 Enter a message → 🔍 Analyze → 🚨 Spam / ✅ Not Spam

---

## 🚀 Features

- 📩 Classifies SMS & email messages
- 🧹 NLP-based text preprocessing
- 🔤 Tokenization & stemming
- 🚫 Stopword removal
- 📊 TF-IDF feature extraction
- 🤖 Multinomial Naive Bayes classification
- 📈 Prediction confidence
- 🌙 Clean dark-themed Streamlit interface
- ⚡ Instant predictions

---

## 🧠 How It Works

```text
        📩 Input Message
               │
               ▼
       🧹 Text Preprocessing
               │
       ┌───────┴────────┐
       │                │
   Tokenization    Stopword Removal
       │                │
       └───────┬────────┘
               ▼
            Stemming
               │
               ▼
         📊 TF-IDF Vectorization
               │
               ▼
      🤖 Multinomial Naive Bayes
               │
               ▼
      ┌────────┴────────┐
      │                 │
 🚨 Spam            ✅ Not Spam

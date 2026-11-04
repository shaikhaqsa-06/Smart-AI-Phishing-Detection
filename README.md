# 🛡️ Smart AI Phishing URL Blocker

AI-powered phishing URL detection system using Machine Learning and Flask.

## 📌 Project Overview

Smart AI Phishing URL Blocker is a machine-learning-based web application that analyzes URL characteristics and predicts whether a URL is potentially phishing or legitimate.

The system uses a trained Random Forest classifier and provides a prediction confidence score through a simple Flask web interface.

## 🤖 Machine Learning Model

Model Used:

- Random Forest Classifier
- Tuned Random Forest model
- Scikit-learn

## 🔍 Features

The system analyzes 30 URL-related features, including:

- IP address usage
- URL length
- URL shortening service
- @ symbol
- Double slash redirecting
- Prefix/Suffix
- Subdomain
- SSL state
- HTTPS token
- Domain-related features
- Webpage-related features

## 📊 Model Performance

The tuned Random Forest model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 94.27% |
| Precision | 93.54% |
| Recall | 94.70% |
| F1 Score | 94.12% |
| ROC-AUC | 99.05% |

## 🌐 Web Application

The application is developed using Flask.

The user enters a URL and the system returns:

- Phishing or legitimate classification
- Prediction confidence
- URL risk information

## 🧪 Example Tests

### Legitimate URL

`https://www.google.com`

Prediction:

**LEGITIMATE URL**

Confidence: **69.23%**

### Phishing Test URL

`http://192.168.1.1/login@google.com`

Prediction:

**PHISHING URL DETECTED**

Confidence: **68.04%**

## 📁 Project Structure

```text
Smart_AI_Phishing_URL_Blocker/
│
├── app.py
├── requirements.txt
├── tuned_random_forest_v2.pkl
└── tuned_random_forest.pkl
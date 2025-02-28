# IMDB Sentiment Analysis using Machine Learning

## Overview

This project applies multiple machine learning models to analyze sentiment in IMDB movie reviews. It includes preprocessing, feature extraction using TF-IDF, and model training using various classifiers.

## Dataset

- **Source**: IMDB Dataset of 50K Movie Reviews [Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Structure**: 50,000 movie reviews labeled as positive or negative

## Preprocessing Steps

- Removed HTML tags and special characters
- Converted text to lowercase
- Tokenized and lemmatized words
- Removed stopwords
- Applied TF-IDF vectorization

## Exploratory Data Analysis (EDA)

- **Class Distribution**: Balanced between positive and negative reviews.
- **Word Cloud**: Displays frequently used words.
- **Sentiment Distribution**: Histogram showing sentiment polarity.

## Machine Learning Models Used

### 1. **Logistic Regression**
- Regularization: L2
- Solver: liblinear
- Accuracy: **~87%**

### 2. **Support Vector Classifier (SVM)**
- Kernel: Linear
- Regularization (C): 0.1
- Accuracy: **~88%**

### 3. **Random Forest Classifier**
- 100 trees, max depth = 5
- Accuracy: **~85%**

### 4. **XGBoost Classifier**
- Learning rate: 0.1, Max depth: 3
- Accuracy: **~89%**

### 5. **LightGBM Classifier**
- Learning rate: 0.1, Num leaves: 31
- Accuracy: **~89%**

## Hyperparameter Tuning (GridSearchCV)

Performed hyperparameter tuning for:
- **Random Forest**
- **XGBoost**
- **LightGBM**

Best parameters were selected based on accuracy and F1-score.

## Ensemble Model (Voting Classifier)

- Combined **Random Forest, XGBoost, and LightGBM** for better predictions.
- Final ensemble accuracy: **~90%**.

## Evaluation Metrics

1. **Accuracy**: Measures correct classifications
2. **F1-Score**: Balance between precision and recall
3. **Classification Report**: Provides precision, recall, and F1-score for each model

## Results & Insights

- **XGBoost and LightGBM performed the best** among individual models.
- **Ensemble learning further improved accuracy** by combining strong classifiers.
- **TF-IDF proved effective** in feature extraction for text classification.

## Installation & Usage

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm nltk matplotlib
```
2. Download the dataset and unzip it.

3. Run the sentiment_analysis.py script to preprocess and train models.

4. Evaluate the model performance using accuracy and classification reports.

### Conclusion

Conclusion
This project demonstrates the effectiveness of various machine learning models in sentiment analysis. Future work could include deep learning techniques like LSTMs and transformers for even better performance.
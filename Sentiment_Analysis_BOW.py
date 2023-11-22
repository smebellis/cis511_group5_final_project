import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import re

# Getting stopwords and punctuations
nltk.download('stopwords')
stopwords_list = set(stopwords.words('english'))
punctuations = string.punctuation

# Reading train and test data file
train_data = pd.read_csv("train_sentiment_dataset.csv")
test_data = pd.read_csv("test_sentiment_dataset.csv")

# Variable initializations
sentiment_mapping = {0: 'negative', 1: 'positive'}
positive_word_counts = {}
negative_word_counts = {}
original_test_labels={}
predicted_test_labels = {}

# Creating Bag-of-words for positive and negative words in train dataset
for index, row in train_data.iterrows():
    sentiment = sentiment_mapping[row['label']]
    text = row['text']
    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        if word not in stopwords_list:
            if sentiment == 'positive':
                if word not in positive_word_counts:
                    positive_word_counts[word] = 1
                else:
                    positive_word_counts[word] += 1
            else:
                if word not in negative_word_counts:
                    negative_word_counts[word] = 1
                else:
                    negative_word_counts[word] += 1

# Predicting sentiments for test data using Bag-of-words from trained data
for index, row in test_data.iterrows():
    sentiment = sentiment_mapping[row['label']]
    text = row['text']
    label = row['label']
    words = re.findall(r'\b\w+\b', text.lower())
    words = [word for word in words if word not in stopwords_list]
    original_test_labels[tuple(words)] = label
    positive_count = sum(positive_word_counts.get(word, 0) for word in words)
    negative_count = sum(negative_word_counts.get(word, 0) for word in words)

    if positive_count > negative_count:
        predicted_sentiment = 1
    else:
        predicted_sentiment = 0
    predicted_test_labels[tuple(words)] = predicted_sentiment

# Calculating the accuracy for Bad-of-words model
correct_predictions = 0
total_predictions = len(original_test_labels)

for words, predicted_sentiment in predicted_test_labels.items():
    original_label = original_test_labels.get(words)
    if original_label is not None and original_label == predicted_sentiment:
        correct_predictions += 1

accuracy = (correct_predictions / total_predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")
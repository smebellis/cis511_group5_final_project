import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import re

nltk.download('stopwords')

data = pd.read_csv("train_sentiment_dataset.csv")

sentiment_mapping = {0: 'negative', 1: 'positive'}

stopwords_list = set(stopwords.words('english'))
punctuations = string.punctuation

positive_word_counts = {}
negative_word_counts = {}

for index, row in data.iterrows():
    sentiment = sentiment_mapping[row['label']]
    text = row['text']
    words = re.findall(r'\b\w+\b', text.lower())  # Extract valid words using regex
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

test_data = pd.read_csv("test_sentiment_dataset.csv")
# test_results = []
# test_words_OG_label=[]
test_results = {}
test_words_OG_label={}


for index, row in test_data.iterrows():
    sentiment = sentiment_mapping[row['label']]
    text = row['text']
    label = row['label']
    words = re.findall(r'\b\w+\b', text.lower())  # Extract valid words using regex
    words = [word for word in words if word not in stopwords_list]
    test_words_OG_label[tuple(words)] = label
    # test_words_OG_label.append((words,label))

    positive_count = sum(positive_word_counts.get(word, 0) for word in words)
    negative_count = sum(negative_word_counts.get(word, 0) for word in words)

    if positive_count > negative_count:
        predicted_sentiment = 1
    else:
        predicted_sentiment = 0

    # test_results.append((words, predicted_sentiment))
    test_results[tuple(words)] = predicted_sentiment

correct_predictions = 0
total_predictions = len(test_words_OG_label)

for words, predicted_sentiment in test_results.items():
    original_label = test_words_OG_label.get(words)
    if original_label is not None and original_label == predicted_sentiment:
        correct_predictions += 1

accuracy = (correct_predictions / total_predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")
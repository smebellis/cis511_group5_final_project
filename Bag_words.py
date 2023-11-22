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
print("Negative words:",negative_word_counts)  #Negative words: {'story': 401, 'line': 90, 'rehashed': 1}
print("positive words:",positive_word_counts)  #positive words: {'joy': 41, 'happy': 900, 'together': 101}


test_data = pd.read_csv("test_sentiment_dataset.csv")
test_words_OG_label ={}
for index, row in test_data.iterrows():
    sentiment = sentiment_mapping[row['label']]
    text = row['text']
    label = row['label']
    words = re.findall(r'\b\w+\b', text.lower())# Extract valid words using regex
    words = [word for word in words if word not in stopwords_list]
    test_words_OG_label[tuple(words)] = label
# print(test_words_OG_label)   #{('qt', 'user', 'original', 'battle', 'of', 'hogwarts', 'happybirthdayremuslupin'): 1}



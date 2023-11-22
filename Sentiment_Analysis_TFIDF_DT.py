from collections import Counter
import math

import csv
import re
import nltk
from nltk import DecisionTreeClassifier
from nltk.corpus import stopwords
import pandas as pd
from sklearn.metrics import confusion_matrix

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import os.path

# Step 1: Implement TF-IDF
def calculate_tf(text):
    words = text.split()
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    max_freq = max(word_freq.values())
    tf = {word: freq / max_freq for word, freq in word_freq.items()}
    return tf


def calculate_idf(documents):
    word_doc_count = {}
    for doc in documents:
        words = set(doc.split())
        for word in words:
            word_doc_count[word] = word_doc_count.get(word, 0) + 1
    idf = {word: math.log(len(documents) / count) for word, count in word_doc_count.items()}
    return idf


def calculate_tfidf(tf, idf):
    return {word: tf[word] * idf.get(word, 0) for word in tf}


# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')


# Function to clean text
def clean_text(text):
    # Remove special symbols and links
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'br', ' ', text)
    # Remove continuous punctuation
    text = re.sub(r'(\W)\1+', r'\1', text)
    # Remove stopwords using NLTK
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)


# Step 2: Read and Prepare Data
def read_csv(file_path):
    texts = []
    labels = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            cleaned_text = clean_text(row[0])
            texts.append(cleaned_text)
            labels.append(int(row[1]))
    return texts, labels


train_texts, train_labels = read_csv('Train_sentiment_dataset.csv') #read csv
test_texts, test_labels = read_csv('test_sentiment_dataset.csv')

# Step 3: Calculate TF-IDF Vectors
train_idf = calculate_idf(train_texts)
train_tfidf_vectors = []
for text in train_texts:
    tf = calculate_tf(text)
    tfidf = calculate_tfidf(tf, train_idf)
    train_tfidf_vectors.append(tfidf)

test_tfidf_vectors = []
for text in test_texts:
    tf = calculate_tf(text)
    tfidf = calculate_tfidf(tf,train_idf)  # Use the IDF from training data This ensures consistency in the IDF values used for both training and test data, preventing errors related to missing IDF values for test data words.
    test_tfidf_vectors.append(tfidf)


# Convert TF-IDF vectors into feature sets
def tfidf_to_features(tfidf_vector):
    return dict(zip(tfidf_vector.keys(), tfidf_vector.values()))

train_features = [(tfidf_to_features(vector), label) for vector, label in zip(train_tfidf_vectors, train_labels)]
test_features = [(tfidf_to_features(vector), label) for vector, label in zip(test_tfidf_vectors, test_labels)]

# combine all cleaned text into one string
wc_training_texts = ' '.join(train_texts)

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=400).generate(wc_training_texts)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Top 10 words histogram
top_words = Counter(wc_training_texts.split()).most_common(10)
top_words_df = pd.DataFrame(top_words, columns=['word', 'freq'])
plt.figure(figsize=(12, 15))
sns.barplot(x="freq", y="word", data=top_words_df)
plt.title(f"Top {10} words in tweets")
plt.show()

# check if the file exists
if os.path.isfile('sentiment_classifier.pickle'):
    print("File exists...loading the classifier\n")
    with open('sentiment_classifier.pickle', 'rb') as file:
        # Load the classifier
        clf = pickle.load(file)
else:
    print("Model does not exist...training the classifier\n")
    with open('sentiment_classifier.pickle', 'wb') as file:
        # Train Decision Tree Classifier
        clf = DecisionTreeClassifier.train(train_features)
        print("saving the classifier...\n")
        pickle.dump(clf, file)
        
# Confusion Matrix
predicted_labels = [clf.classify(feats) for feats, _ in test_features]

cm = confusion_matrix(test_labels, predicted_labels, labels=[1, 0])
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'], cmap='cividis')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion matrix for Sentiment Analysis TFIDF DT')
plt.show()

# Evaluate the classifier on test data
accuracy = nltk.classify.accuracy(clf, test_features) * 100
print(f"Accuracy: {accuracy:.2f}%")

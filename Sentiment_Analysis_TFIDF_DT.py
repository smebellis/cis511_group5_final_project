import math
import csv
import re
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

nltk.download('stopwords')


# Function for calculating TF
def calculate_tf(text):
    words = text.split()
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    max_freq = max(word_freq.values())
    tf = {word: freq / max_freq for word, freq in word_freq.items()}
    return tf


# Function for calculating IDF
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


# Cleaning data by removing stopwords and special symbols
def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'(\W)\1+', r'\1', text)
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)


# Reading train and test datasets
def read_csv(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            cleaned_text = clean_text(row[0])
            texts.append(cleaned_text)
            labels.append(int(row[1]))
    return texts, labels


train_texts, train_labels = read_csv('train_sentiment_dataset.csv')
test_texts, test_labels = read_csv('test_sentiment_dataset.csv')

# Calculating TF-IDF Vectors for train and test data
train_idf = calculate_idf(train_texts)
train_tfidf_vectors = []
for text in train_texts:
    tf = calculate_tf(text)
    tfidf = calculate_tfidf(tf, train_idf)
    train_tfidf_vectors.append(tfidf)

test_tfidf_vectors = []
for text in test_texts:
    tf = calculate_tf(text)
    tfidf = calculate_tfidf(tf, train_idf)
    test_tfidf_vectors.append(tfidf)


# Creating Feature vector as TF-IDF values
def tfidf_to_features(tfidf_vector):
    return dict(zip(tfidf_vector.keys(), tfidf_vector.values()))


train_features = [tfidf_to_features(vector) for vector in train_tfidf_vectors]
test_features = [tfidf_to_features(vector) for vector in test_tfidf_vectors]

# Training decision tree model and predicting labels for test data
vectorizer = DictVectorizer(sparse=False)
X_train = vectorizer.fit_transform(train_features)
X_test = vectorizer.transform(test_features)
y_train = train_labels
y_test = test_labels

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions) * 100
print(f"Accuracy: {accuracy:.2f}%")
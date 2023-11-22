import csv
import math


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
    return {word: tf[word] * idf[word] for word in tf}


# Step 2: Read and Prepare Data
def read_csv(file_path):
    texts = []
    labels = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            texts.append(row[0])
            labels.append(int(row[1]))
    return texts, labels


train_texts, train_labels = read_csv('Train_sentiment_dataset.csv')
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
    tfidf = calculate_tfidf(tf, train_idf)  # Use the IDF from training data
    test_tfidf_vectors.append(tfidf)

print(test_tfidf_vectors)
from collections import Counter
import math
import csv
import re
import json
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


# Convert TF-IDF vectors into feature sets
def tfidf_to_features(tfidf_vector):
    return dict(zip(tfidf_vector.keys(), tfidf_vector.values()))



def display_wordcloud(train_texts):
    
    wc_training_texts = ' '.join(train_texts)
    # Generate a word cloud image
    wordcloud = WordCloud(width=800, height=400).generate(wc_training_texts)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    
def display_histogram(train_texts):
    
    wc_training_texts = ' '.join(train_texts)
    # Top 10 words histogram
    top_words = Counter(wc_training_texts.split()).most_common(10)
    top_words_df = pd.DataFrame(top_words, columns=['word', 'freq'])
    plt.figure(figsize=(12, 15))
    sns.barplot(x="freq", y="word", data=top_words_df)
    plt.title(f"Top {10} words in tweets")
    plt.show()

def check_model_exists(filename):
    
    # check if the file exists
    if os.path.isfile(f'{filename}'):
        print("File exists...loading the classifier\n")
        with open(f'{filename}', 'rb') as file:
            # Load the classifier
            clf = pickle.load(file)
    else:
        print("Model does not exist...training the classifier\n")
        with open(f'{filename}', 'wb') as file:
            # Train Decision Tree Classifier
            clf = DecisionTreeClassifier.train(train_features)
            print('Classifier trained successfully...\n')
            pickle.dump(clf, file)
            print("Classifier Saved successfully...\n")
        
    return clf

def calculate_most_frequent_class_baseline(labels):
    # Count the frequency of each label
    label_counts = Counter(labels)

    # Find the most common label
    most_common_label, most_common_count = label_counts.most_common(1)[0]

    # Calculate the baseline accuracy: (number of times most common label appears) / (total number of labels)
    baseline_accuracy = (most_common_count / len(labels)) * 100

    return baseline_accuracy, most_common_label



# Display Confusion Matrix
def display_confusion_matrix(test_labels, test_features, clf):
    predicted_labels = [clf.classify(feats) for feats, _ in test_features]
    cm = confusion_matrix(test_labels, predicted_labels, labels=[1, 0])
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'], cmap='cividis')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix for Sentiment Analysis TFIDF DT')
    plt.show()


def display_model_accuracy(improvement, baseline_accuracy, random_guess_accuracy, model_accuracy):

    # Data to be plotted
    accuracies = [baseline_accuracy, random_guess_accuracy, model_accuracy]
    labels = ['Most Common Class', 'Random Guess', 'Model Accuracy']

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, accuracies, color=['blue', 'green', 'red'])
    plt.xlabel('Method')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Model Accuracy with Baselines')
    plt.ylim(0, 100)  # Set y-axis range for clarity

    # Annotating the improvement
    plt.annotate(f'Improvement: {improvement:.2f}%', 
                xy=('Model Accuracy', model_accuracy), 
                xytext=(1, model_accuracy+5),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                ha='center')

    plt.show()


def main():
    # Download NLTK stopwords if not already downloaded
    nltk.download('stopwords')
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

    train_features = [(tfidf_to_features(vector), label) for vector, label in zip(train_tfidf_vectors, train_labels)]
    test_features = [(tfidf_to_features(vector), label) for vector, label in zip(test_tfidf_vectors, test_labels)]

    # Step 4: Train and Evaluate Classifier
    clf = check_model_exists('sentiment_classifier.pickle') 

    # Step 5: Evaluate the classifier on test data
    baseline_accuracy, most_common_label = calculate_most_frequent_class_baseline(test_labels)
    print(f"Most common label: {most_common_label}")
    print(f"Most Common Class: {baseline_accuracy:.2f}%")

    random_guess_accuracy = 50  # Assuming a baseline (like random guessing)
    print(f"Random Guess Accuracy: {random_guess_accuracy}%")

    accuracy = nltk.classify.accuracy(clf, test_features) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    improvement = accuracy - baseline_accuracy
    print(f"Improvement Over Baseline: {improvement:.2f}%")

    display_model_accuracy(improvement, baseline_accuracy, random_guess_accuracy, accuracy)
    display_wordcloud(train_texts)
    display_histogram(train_texts)
    display_confusion_matrix(test_labels, test_features, clf)
    
    model_name = 'Sentiment_Analysis_TFIDF_DT'
    with open(f'{model_name}_accuracy.json', 'w') as file:
        json.dump({'model': model_name, 'accuracy': accuracy}, file)

if __name__ == "__main__":
    main()

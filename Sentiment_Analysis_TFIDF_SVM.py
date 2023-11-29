import math
import csv
import re
import nltk
import Sentiment_Analysis_TFIDF_DT as dt
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    nltk.download('stopwords')

    train_texts, train_labels = dt.read_csv('train_sentiment_dataset.csv')
    test_texts, test_labels = dt.read_csv('test_sentiment_dataset.csv')

    train_idf = dt.calculate_idf(train_texts)
    train_tfidf_vectors = []
    for text in train_texts:
        tf = dt.calculate_tf(text)
        tfidf = dt.calculate_tfidf(tf, train_idf)
        train_tfidf_vectors.append(tfidf)

    test_tfidf_vectors = []
    for text in test_texts:
        tf = dt.calculate_tf(text)
        tfidf = dt.calculate_tfidf(tf, train_idf)
        test_tfidf_vectors.append(tfidf)


    train_features = [dt.tfidf_to_features(vector) for vector in train_tfidf_vectors]
    test_features = [dt.tfidf_to_features(vector) for vector in test_tfidf_vectors]

    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(train_features)
    X_test = vectorizer.transform(test_features)
    y_train = train_labels
    y_test = test_labels

    svmClf = make_pipeline(StandardScaler(), SVC(kernel='poly',gamma='auto'))
    svmClf.fit(X_train, y_train)


    predictionsSVM = svmClf.predict(X_test)
    accuracySVM = accuracy_score(y_test, predictionsSVM) * 100
    print(f"Accuracy SVM: {accuracySVM:.2f}%")

if __name__ == "__main__":
    main()
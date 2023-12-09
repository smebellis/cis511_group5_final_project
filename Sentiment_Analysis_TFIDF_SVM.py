import json
import nltk
import Sentiment_Analysis_TFIDF_DT as dt
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

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

svmClf = make_pipeline(StandardScaler(), SVC(kernel='poly', gamma='auto'))
svmClf.fit(X_train, y_train)

predictionsSVM = svmClf.predict(X_test)
accuracySVM = accuracy_score(y_test, predictionsSVM) * 100
print(f"Accuracy SVM: {accuracySVM:.2f}%")


def plot_accuracy_comparison(y_train, y_test, predictions, title='Accuracy Comparison'):
    # Baseline 1: Random Classifier
    dummy_clf_random = DummyClassifier(strategy="uniform")
    dummy_clf_random.fit(y_train, y_train)
    random_accuracy = dummy_clf_random.score(y_test, y_test) * 100

    # Baseline 2: Majority Class Classifier
    dummy_clf_majority = DummyClassifier(strategy="most_frequent")
    dummy_clf_majority.fit(y_train, y_train)
    majority_accuracy = dummy_clf_majority.score(y_test, y_test) * 100

    # model's accuracy
    model_accuracy = accuracy_score(y_test, predictions) * 100

    # Data for plotting
    labels = ['Random Classifier', 'Majority Class', 'SVM Model']
    accuracies = [random_accuracy, majority_accuracy, model_accuracy]

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(labels, accuracies, color=['blue', 'green', 'red'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)

    # Display the accuracies on the graph
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.5, f"{v:.2f}%", color='black', ha='center')

    plt.show()


model_name = 'SVM'
with open(f'{model_name}_accuracy.json', 'w') as file:
    json.dump({'model': model_name, 'accuracy': accuracySVM}, file)

if __name__ == "__main__":
    plot_accuracy_comparison(y_train, y_test, predictionsSVM, title='Baseline vs SVM Model Accuracy')

import pandas as pd
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

data = pd.read_csv("tweet_eval_sentiment_dataset.csv")

sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

stopwords_list = set(stopwords.words('english'))

word_counts = {sentiment: defaultdict(int) for sentiment in sentiment_mapping.values()}

for index, row in data.iterrows():
    sentiment = sentiment_mapping[row['label']]
    text = row['text']
    words = text.split()
    words = [word for word in words if word.lower() not in stopwords_list]
    for word in words:
        word_counts[sentiment][word] += 1

for sentiment, counts in word_counts.items():
    print(f"Words for {sentiment} :")
    print(counts)

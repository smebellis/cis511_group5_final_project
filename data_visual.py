import unittest
import token
import numpy as np
import pandas as pd
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from PIL import Image
import re
from bs4 import BeautifulSoup
import lxml
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

# create a list of stopword to remove
stop_words = stopwords.words('english')

df = pd.read_csv("tweet_eval_sentiment_dataset.csv")

sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

pd.options.display.max_colwidth = 200
class TestCleaningTweets(unittest.TestCase):

    def test_remove_html(self):
        tweet = "<html>This is a <b>tweet</b></html>"
        cleaned = cleaning_tweets(tweet)
        self.assertNotIn('<', cleaned)
        self.assertNotIn('>', cleaned)

    def test_to_lower(self):
        tweet = "This is a TWEET"
        cleaned = cleaning_tweets(tweet)
        self.assertEqual(cleaned, cleaned.lower())

    def test_remove_mentions_and_emoticons(self):
        tweet = "@user Hello :) #exciting"
        cleaned = cleaning_tweets(tweet)
        self.assertNotIn("@user", cleaned)
        self.assertNotIn(":)", cleaned)
        self.assertNotIn("#exciting", cleaned)

    def test_tokenize_and_remove_short_words(self):
        tweet = "This is a tweet with some words"
        cleaned = cleaning_tweets(tweet)
        for word in cleaned.split():
            self.assertGreaterEqual(len(word), 3)
            
    def lower_case(self):
        tweet = "This is a TWEET"
        cleaned = cleaning_tweets(tweet)
        self.assertEqual(cleaned, cleaned.lower())

#############################################
#####    REGEX PATTERNS FOR CLEANING    #####
#############################################

#remove the emoticons and symbols from tweets
regex_pattern =  re.compile(pattern= "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    "]+", flags = re.UNICODE)

# remove the urls from tweets
url_pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')

# remove @mentions and hashes from the text
re_list = ['@[A-Za-z0-9_]+', '#']
combined_re = re.compile( '|'.join( re_list) )



def cleaning_tweets(t):
    # remove the html tags from tweets
    del_amp = BeautifulSoup(t, 'lxml')
    del_amp_text = del_amp.get_text()
    remove_mentions = re.sub(combined_re, '', del_amp_text)
    remove_emoticons = re.sub(regex_pattern, '', remove_mentions)
    lower_case = remove_emoticons.lower()
    tokenize_words = word_tokenize(lower_case)
    result_words = [x for x in tokenize_words if len(x) > 2]
    return (" ".join(result_words)).strip()


print("Cleaning the tweets...\n")
cleaned_tweets = []
for i, tweet in enumerate(df['text']):
    if((i+1) % 100 == 0):
        print(f"Tweets processed: {i+1} of {len(df)}")
    cleaned_tweets.append(cleaning_tweets(df['text'][i]))
    

string = pd.Series(cleaned_tweets).str.cat(sep=' ')


# create a word cloud image
wordcloud = WordCloud(width=800, height=800,
    collocations=False,
    background_color='white',
    stopwords=stop_words,
    min_font_size=10).generate(string)
# plot the word cloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Top 50 words histogram
top_words = Counter(string.split()).most_common(10)
top_words_df = pd.DataFrame(top_words, columns=['word', 'freq'])
plt.figure(figsize=(12, 15))
sns.barplot(x="freq", y="word", data=top_words_df)
plt.title(f"Top {10} words in tweets")
plt.show()

# Add a Confusion Matrix to display the sentiment of the tweets
# cm = confusion_matrix()

# A heatmap to show the correlation between the sentiment and the text
# fig, ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"])
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion matrix for Sentiment Analysis')
# plt.show()

# https://dennisalexandermorozov.medium.com/sentiment-analysis-of-twitter-text-and-visualization-methods-d6ab365ccfbc


# https://towardsdatascience.com/nlp-text-preprocessing-steps-tools-and-examples-94c91ce5d30Ad

if __name__ == '__main__':
    
    unittest.main()
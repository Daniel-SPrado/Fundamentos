import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

plt.style.use('fivethirtyeight')

cols = ['sentiment','id','date','query_string','user','text']

df = pd.read_csv("./trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None, names=cols)
df.head()
df.info()
df.sentiment.value_counts()

df.query_string.value_counts()
df.drop(['id','date','query_string','user'],axis=1,inplace=True)
df.head()

df[df.sentiment == 0].head(10)
df[df.sentiment == 4].head(10)
df[df.sentiment == 0].index
df[df.sentiment == 4].index
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
df.sentiment.value_counts()
df['pre_clean_len'] = [len(t) for t in df.text]
from pprint import pprint
data_dict = {
    'sentiment':{
        'type':df.sentiment.dtype,
        'description':'sentiment class - 0:negative, 1:positive'
    },
    'text':{
        'type':df.text.dtype,
        'description':'tweet text'
    },
    'pre_clean_len':{
        'type':df.pre_clean_len.dtype,
        'description':'Length of the tweet before cleaning'
    },
    'dataset_shape':df.shape
}

pprint(data_dict)

fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()

df[df.pre_clean_len > 140].head(10)

df.text[279]


example1: BeautifulSoup(df.text[279], 'lxml')
#print example1.get_text()

df.text[343]

import re
re.sub(r'@[A-Za-z0-9]+','',df.text[343])

df.text[0]

re.sub('https?://[A-Za-z0-9./]+','',df.text[0])

df.text[226]

testing = df.text[226].decode("utf-8-sig")
testing


testing.replace(u"\ufffd", "?")

df.text[175]

re.sub("[^a-zA-Z]", " ", df.text[175])

from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()


testing = df.text[:100]

test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))

test_result

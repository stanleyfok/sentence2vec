import pandas as pd
import re
import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# pre processing data
def cleanData(sentence):
    # convert to lowercase, ignore all special characters - keep only
    # alpha-numericals and spaces
    sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())

    # remove stop words
    sentence = " ".join([word for word in sentence.split()
                        if word not in stopwords.words('english')])

    return sentence


df = pd.read_csv('./data/job_titles.csv', encoding='ISO-8859-1', index_col=0)

# drop duplicate rows
df = df.drop_duplicates(subset='title')

# clean data
df['title'] = df['title'].map(lambda x: cleanData(x))

# get array of titles
titles = df['title'].values.tolist()

# tokenize the each title
tok_titles = [word_tokenize(title) for title in titles]

# refer to here for all parameters:
# https://radimrehurek.com/gensim/models/word2vec.html
model = Word2Vec(tok_titles, sg=1, size=100, window=5, min_count=5, workers=4,
                 iter=100)

# save model to file
model.save('./data/job_titles.model')

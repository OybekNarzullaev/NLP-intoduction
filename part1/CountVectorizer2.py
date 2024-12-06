import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

from pathlib import Path
BASE_DIR = Path.cwd()
df = pd.read_csv(f"{BASE_DIR}/data/bbc_text_cls.csv")

print(df.head())

inputs = df['inputs']
labels = df['labels']

labels.hist(figsize=(10,5))
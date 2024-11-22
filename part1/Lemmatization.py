import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Sample words to lemmatize
words = ["running", "ran", "runs", "better", "houses"]

# Lemmatize without specifying part of speech (default is noun)
print("Default Lemmatization (Noun):")
print([lemmatizer.lemmatize(word) for word in words])

# Lemmatize with part of speech specified
print("\nLemmatization with Part of Speech:")
print([lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words])

# Custom
print(lemmatizer.lemmatize("was", pos=wordnet.VERB)) # be
print(lemmatizer.lemmatize("going")) # going
print(lemmatizer.lemmatize("going", pos=wordnet.VERB)) # go
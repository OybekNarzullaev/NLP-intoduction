from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# Initialize the stemmers
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")  # Specify the language

# Sample words to stem
words = ["running", "ran", "runner", "easily", "fairly"]

# Apply stemming with different stemmers
print("Porter Stemmer Results:")
print([porter.stem(word) for word in words])

print("\nLancaster Stemmer Results:")
print([lancaster.stem(word) for word in words])

print("\nSnowball Stemmer Results:")
print([snowball.stem(word) for word in words])

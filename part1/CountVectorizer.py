from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents1 = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Initialize CountVectorizer
vectorizer_by_word = CountVectorizer()

# Fit and transform the documents
X = vectorizer_by_word.fit_transform(documents1)

# View the feature names (vocabulary)
print("Vocabulary:", vectorizer_by_word.get_feature_names_out())

# View the document-term matrix
print("Document-term matrix:\n", X.toarray())


# Sample documents
documents2 = [
    "hello world",
    "machine learning",
    "hello machine"
]

# Initialize character-based CountVectorizer
vectorizer_by_char = CountVectorizer(analyzer='char', ngram_range=(3, 3))  # Trigrams

# Fit and transform the documents
X = vectorizer_by_char.fit_transform(documents2)

# Display feature names (character trigrams)
print("Character n-grams (vocabulary):", vectorizer_by_char.get_feature_names_out())

# Display the document-term matrix
print("Document-term matrix:\n", X.toarray())
import pandas as pd

df = pd.read_csv(
    r"C:\Users\L\Downloads\Mobile Devices\twitter_training.csv",
    header=None
)

print(df.head())


# ==========================================
# 1. Import Libraries
# ==========================================

import pandas as pd
import numpy as np
import re
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# ==========================================
# 2. Download NLTK Resources
# ==========================================

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# ==========================================
# 3. Load Dataset
# ==========================================

df = pd.read_csv(
    r"C:\Users\L\Downloads\Mobile Devices\twitter_training.csv",
    header=None
)

df.columns = ["Tweet_ID", "Entity", "Sentiment", "Tweet"]

print("\nOriginal Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ==========================================
# 4. Data Inspection
# ==========================================

print("\nDataset Information:")
print(df.info())

print("\nSentiment Distribution:")
print(df["Sentiment"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 5. Handle Missing Values
# ==========================================

df = df.dropna(subset=["Tweet", "Sentiment"])

print("\nAfter Removing Missing Tweets:")
print(df.shape)


# ==========================================
# 6. Remove Duplicate Rows
# ==========================================

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape After Removing Duplicates:")
print(df.shape)


# ==========================================
# 7. Preserve Original Tweets
# ==========================================

df["Original_Tweet"] = df["Tweet"].astype(str)


# ==========================================
# 8. Contraction Expansion
# ==========================================

contractions = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "didn't": "did not",
    "doesn't": "does not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "i've": "i have",
    "they've": "they have",
    "i'll": "i will"
}

def expand_contractions(text):

    for word, replacement in contractions.items():

        text = re.sub(
            r"\b" + re.escape(word) + r"\b",
            replacement,
            text
        )

    return text


# ==========================================
# 9. Text Cleaning Function
# ==========================================

def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove user mentions
    text = re.sub(r"@\w+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Preserve hashtag words
    text = text.replace("#", "")

    # Expand contractions
    text = expand_contractions(text)

    # Convert emojis into text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["Cleaned_Tweet"] = df["Original_Tweet"].apply(clean_text)


# ==========================================
# 10. Tokenization
# ==========================================

df["Tokens"] = df["Cleaned_Tweet"].apply(
    lambda text: text.split()
)


# ==========================================
# 11. Stopword Removal
# ==========================================

stop_words = set(stopwords.words("english"))

# Preserve negation words for sentiment analysis
stop_words = stop_words - {"not", "no", "nor"}

df["Tokens_No_Stopwords"] = df["Tokens"].apply(
    lambda words: [
        word for word in words
        if word not in stop_words
    ]
)


# ==========================================
# 12. Stemming
# ==========================================

stemmer = PorterStemmer()

df["Stemmed_Tweet"] = df["Tokens_No_Stopwords"].apply(
    lambda words: [
        stemmer.stem(word) for word in words
    ]
)


# ==========================================
# 13. Lemmatization
# ==========================================

lemmatizer = WordNetLemmatizer()

df["Lemmatized_Tweet"] = df["Tokens_No_Stopwords"].apply(
    lambda words: [
        lemmatizer.lemmatize(word) for word in words
    ]
)


# ==========================================
# 14. Convert Tokens Back Into Text
# ==========================================

df["Final_Tweet"] = df["Lemmatized_Tweet"].apply(
    lambda words: " ".join(words)
)


# ==========================================
# 15. Final Dataset Inspection
# ==========================================

print("\nOriginal vs Cleaned Tweets:")

print(
    df[
        ["Original_Tweet", "Cleaned_Tweet", "Final_Tweet"]
    ].head(10).to_string(index=False)
)

print("\nFinal Dataset Shape:")
print(df.shape)


# ==========================================
# 16. Save Preprocessed Dataset
# ==========================================

df.to_csv(
    "twitter_preprocessed.csv",
    index=False
)

print("\nPreprocessing Completed Successfully!")
print("File Saved: twitter_preprocessed.csv")

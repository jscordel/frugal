import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd

def nltk_downloads():
#Download necessary NLTK data files
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def pre_clean_data(data : pd.DataFrame):

    """ Only clean text in a DataFrame or line by line without preprocessing (no tokennizer no Lemmmentizer )"""
    # Pre-cleaning
    text_cleaned = data.lower()
    text_cleaned = ''.join(char for char in text_cleaned if not char.isdigit())
    for punctuation in string.punctuation:
        text_cleaned = text_cleaned.replace(punctuation, '')
    text_cleaned = text_cleaned.strip()



    return text_cleaned


def preprocess_text(text):
    """
    Preprocess the input text by converting it to lowercase, removing digits and punctuation,
    tokenizing, and lemmatizing.

    Stopwords are intentionally kept to preserve negation.
    """
    nltk_downloads()

    # Pre-cleaning
    text = text.lower()
    text = ''.join(char for char in text if not char.isdigit())
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    text = text.strip()

    # Tokenize: split text into list of words
    text = word_tokenize(text)

    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    return text

# Test:
if __name__ == "__main__":
    sample_text = "Global Warming? Tell that to the southern districts that woke up to negative 10 degrees this morning."
    processed = preprocess_text(sample_text)
    print(processed)

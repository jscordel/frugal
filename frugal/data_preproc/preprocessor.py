import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

def nltk_downloads():
#Download necessary NLTK data files
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

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

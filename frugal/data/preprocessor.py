import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# à mettre dans une fonction à part dans le dossier data c^ preprocessor.py
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    """
    Preprocess the input text by converting it to lowercase, removing digits and punctuation,
    tokenizing, and lemmatizing.

    Stopwords are intentionally kept to preserve negation.
    """
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

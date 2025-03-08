from datasets import load_dataset
from frugal.config.environment import DATA_DIR
from frugal.data_preproc.preprocessor import preprocess_text
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load_datasets():
    """
    Loads train and test datasets from Hugging Face and returns them as pandas DataFrames.
    """
    train_dataset = pd.DataFrame(load_dataset('QuotaClimat/frugalaichallenge-text-train', split='train', cache_dir=DATA_DIR))
    test_dataset = pd.DataFrame(load_dataset('QuotaClimat/frugalaichallenge-text-train', split='test', cache_dir=DATA_DIR))

    print("✅ Train and test datasets successfully loaded from Hugging Face 🤗")
    return train_dataset, test_dataset

def make_train_test_split(train_dataset, test_dataset):
    """
    Splits datasets into features (quotes) and labels, returning separate DataFrames/Series.
    """
    X_train = train_dataset[['quote']]
    X_test = test_dataset[['quote']]
    y_train = train_dataset['label']
    y_test = test_dataset['label']

    print("✅ Data successfully split into X and y for train and test sets")
    return X_train, X_test, y_train, y_test

def load_and_split_data():
    """
    Loads datasets from source and splits them into features and labels for training and testing.
    """
    train_dataset, test_dataset = load_datasets()
    X_train, X_test, y_train, y_test = make_train_test_split(train_dataset, test_dataset)
    return X_train, X_test, y_train, y_test

def encoding_and_preproc(X_train, X_test, y_train, y_test):
    """
    Applies text preprocessing to features and encodes labels into numeric format.
    """
    label_encoder = LabelEncoder()

    # Apply preprocessing to quotes
    X_train_preproc = pd.DataFrame(X_train['quote'].apply(preprocess_text))
    X_test_preproc = pd.DataFrame(X_test['quote'].apply(preprocess_text))

    # Encode labels into numeric form
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("✅ Features preprocessed and labels encoded successfully")
    return X_train_preproc, X_test_preproc, y_train_encoded, y_test_encoded

# Test
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_preproc, X_test_preproc, y_train_encoded, y_test_encoded = encoding_and_preproc(X_train, X_test, y_train, y_test)

    print("\n🔍 Preview of preprocessed training data (first 3 rows):")
    print(X_train_preproc.head(3))
    print("\nData type of preprocessed training set:", type(X_train_preproc))
    print("\nExample encoded label from y_train_encoded:", y_train_encoded[0])
    print("Data type of encoded labels:", type(y_train_encoded))

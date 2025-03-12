from frugal.data_preproc.data import load_and_split_data , load_datasets ,make_train_test_split
from frugal.data_preproc.preprocessor import pre_clean_data
from frugal.evaluation import evaluate, concatenate_evaluations , classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from datasets import load_dataset
from frugal.config.environment import DATA_DIR
import os
import pandas as pd

def load_and_split_data_withou_class_n(filtered_class : str):
    """
    Loads datasets from source withou a certain class "filter_class" and splits them into features and labels for training and testing.
    """
    train_dataset, test_dataset = load_datasets()

    train_dataset_filtered = train_dataset[train_dataset['label']!=filtered_class]
    test_dataset_filtered = test_dataset[test_dataset['label']!=filtered_class]
    X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = make_train_test_split(train_dataset_filtered,test_dataset_filtered)
    return X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered

""" def pre_clean_data(data : pd.DataFrame):
    def clean_one_quote(quote : str):
        # Pre-cleaning
        text = text.lower()
        text = ''.join(char for char in text if not char.isdigit())
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        text = text.strip()

    data.map(clean_one_quote()) """

def main():
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the name of the folder
    model_name = os.path.basename(script_directory)
    print(model_name)

    # Load datasets and make data lists
    X_train, X_test, y_train , y_test = load_and_split_data_withou_class_n("0_not_relevant")

    #print(len(X_train)+ len(X_test))

    # Clean

    X_train, X_test = X_train.map(pre_clean_data), X_test.map(pre_clean_data)

    # Train and predict with model
    y_pred = tfidf(X_train, X_test, y_train)
    # Get model performance
    evaluate(model_name, y_test, y_pred)
    # Concatenate all evaluations
    concatenate_evaluations()


def tfidf(X_train, X_test, y_train) :
    # Vectorisation
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train.quote)
    X_test_vec = vectorizer.transform(X_test.quote)
    # Training
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    # Prediction
    y_pred = model.predict(X_test_vec)
    return y_pred


if __name__ == "__main__":
    main()

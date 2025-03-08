from frugal.data.data import load_data
from frugal.evaluation import evaluate, concatenate_evaluations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os


def main():
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the name of the folder
    model_name = os.path.basename(script_directory)
    print(model_name)

    # Load datasets and make data lists
    X_train, X_test, y_train, y_test = load_data()
    # Train and predict with model
    y_pred = dummy_model(X_train, X_test, y_train)
    # Get model performance
    evaluate(model_name, y_test, y_pred)
    # Concatenate all evaluations
    concatenate_evaluations()


def dummy_model(X_train, X_test, y_train) :
    # Vectorisation
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    # Training
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    # Prediction
    y_pred = model.predict(X_test_vec)
    return y_pred


if __name__ == "__main__":
    main()

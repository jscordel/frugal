from frugal.data.data import load_data, make_data_lists
from frugal.evaluation import evaluate


def main():
    # Load datasets and make data lists
    X_train, X_test, y_train, y_test = load_data()
    # Train and predict with model
    y_pred = dummy_model(X_train, X_test, y_train)
    # Get model performance
    evaluate(y_test, y_pred)


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

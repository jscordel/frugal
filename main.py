from frugal.data.data import load_data, make_data_lists
from frugal.evaluation import evaluate

##### TO DELETE WHEN REAL MODEL FILE IS DONE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    train_dataset, test_dataset = load_data()
    print("✅ Data loaded from 🤗 ")

    X_train, X_test, y_train, y_test = make_data_lists(train_dataset, test_dataset)
    print("✅ X, y test & train lists made")

    y_pred = dummy_model(X_train, X_test, y_train)

    evaluate(y_test, y_pred)


##### TO DELETE WHEN REAL MODEL FILE IS DONE
def dummy_model(X_train, X_test, y_train) :
    # Vectorisation des textes
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    # Entraînement du modèle
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    # Prédictions
    y_pred = model.predict(X_test_vec)
    return y_pred


if __name__ == "__main__":
    main()

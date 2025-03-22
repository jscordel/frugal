import os
import pickle
import pandas as pd
import numpy as np
from codecarbon import EmissionsTracker
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from frugal.data_preproc.data import load_and_split_data
from frugal.evaluation import evaluate, run_comparisons, confusion_matrix_plot


tracker = EmissionsTracker(log_level="error")

### --- ADD YOUR MODEL HERE & CHANGE THE FUNCTION NAME --- ###
### --- MAKE SURE TO INCLUDE TRAIN & PREDICT STEPS --- ###
def CountVect_MulitinomialNB_train(X_train, X_test, y_train) :
    # Vectorisation
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train['quote'])
    X_test_vec = vectorizer.transform(X_test['quote'])
    # Training
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    ## Save everything required for prediction
    ## ------ REPLACE FILENAMES WITH YOUR MODEL NAME ------ ##
    # -- Save the trained model
    model_filename = "CountVect_MultinomialNB_model.pkl"
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model saved to {model_filename}")
    # -- Save the vectorizer
    vectorizer_filename = "CountVectorizer.pkl"
    with open(vectorizer_filename, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    print(f"Vectorizer saved to {vectorizer_filename}")

    # Prediction
    y_pred = model.predict(X_test_vec)

    return y_pred


def main():
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the name of the folder
    model_name = os.path.basename(script_directory)
    print(model_name)

    # Load datasets and make data lists
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Train and predict with model further defined below.
    # This is what is measured for emissions.
    tracker.start()
    ## ------ REPLACE YOUR MODEL FUNCTION NAME ------ ##
    y_pred = CountVect_MulitinomialNB_train(X_train, X_test, y_train)
    ## ------ REPLACE YOUR MODEL FUNCTION NAME ------ ##
    tracker.stop()
    model_emissions = tracker.final_emissions_data.emissions


    # Evaluate model performance
    if not isinstance(y_test, (pd.Series, np.ndarray)):
        raise TypeError("y_test should be a pandas Series or numpy array")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("y_pred should be a pandas Series or numpy array")
    evaluate(model_name, y_test, y_pred, model_emissions)

    # Run comparisons
    run_comparisons()

    # Confusion matrix of the model
    confusion_matrix_plot(y_test, y_pred)


if __name__ == "__main__":
    main()

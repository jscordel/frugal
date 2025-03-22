from frugal.data_preproc.data import load_and_split_data
from frugal.evaluation import evaluate, concatenate_evaluations, global_model_comparison, global_class_comparison
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import pandas as pd

from codecarbon import EmissionsTracker

tracker = EmissionsTracker(log_level="error")

def main():
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the name of the folder
    model_name = os.path.basename(script_directory)
    print(model_name)

    # Load datasets and make data lists
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Train and predict with model further defined below.
    # THIS IS WHAT IS QUANTIFIED FOR ENERGY CONSUMPTION
    tracker.start()
    y_pred = CountVect_MulitinomialNB_train(X_train, X_test, y_train)
    tracker.stop()
    model_emissions = tracker.final_emissions_data.emissions

    # Get model performance
    # THIS IS NOT QUANTIFIED FOR ENERGY CONSUMPTION BECAUSE NEGLIBLE
    evaluate(model_name, y_test, y_pred, model_emissions)
    # Concatenate all evaluations
    global_data_df = concatenate_evaluations()
    # Global model and class performance comparison
    global_model_comparison(global_data_df)
    global_class_comparison(global_data_df)


#### ADD YOUR MODEL HERE ####
### MAKE SURE TO INCLUDE TRAIN & PREDICT STEPS ###
def CountVect_MulitinomialNB_train(X_train, X_test, y_train) :
    # Vectorisation
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train['quote'])
    X_test_vec = vectorizer.transform(X_test['quote'])
    # Training
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    # Prediction
    y_pred = model.predict(X_test_vec)
    return y_pred


if __name__ == "__main__":
    main()

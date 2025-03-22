import os
import pandas as pd
import numpy as np
from codecarbon import EmissionsTracker
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from frugal.data_preproc.data import load_and_split_data, encoding_and_preproc
from frugal.evaluation import evaluate, run_comparisons, confusion_matrix_plot
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

tracker = EmissionsTracker(log_level="error")

def preprocess_texts(X_train, X_test):
    """Tokenize and pad text data."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train["quote"])

    X_train_token = tokenizer.texts_to_sequences(X_train["quote"])
    X_test_token = tokenizer.texts_to_sequences(X_test["quote"])

    vocab_size = len(tokenizer.word_index)
    max_length = max(len(seq) for seq in X_train_token + X_test_token)

    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post', maxlen=max_length)
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post', maxlen=max_length)
    return X_train_pad, X_test_pad, vocab_size, max_length


def RNN_basic_train(X_train_pad, y_train_encoded, vocab_size, max_length):
    """Define, compile, and train the RNN model."""
    embedding_dim = 50

    # Define the LSTM model
    model = Sequential([
        Masking(mask_value=0., input_shape=(max_length,)),
        Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, mask_zero=True),
        LSTM(64),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    # Train the model
    model.fit(
        X_train_pad,
        y_train_encoded,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping])

    # -- Save the trained model
    model_filename = "RNN_basic_model.keras"
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    return model


def main():
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the name of the folder (model name)
    model_name = os.path.basename(script_directory)
    print(model_name)

    # Load datasets and make data lists
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Train and predict with model further defined below.
    # This is what is measured for emissions.
    tracker.start()

    # Preprocess the data (embedding and padding)
    X_train_preproc, X_test_preproc, y_train_encoded, y_test_encoded = encoding_and_preproc(X_train, X_test, y_train, y_test)
    X_train_pad, X_test_pad, vocab_size, max_length = preprocess_texts(X_train_preproc, X_test_preproc)

    # Train model
    model = RNN_basic_train(X_train_pad, y_train_encoded, vocab_size, max_length)
    #model = load_model("RNN_basic_model.keras")

    y_pred_probs = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_probs, axis=1)

    tracker.stop()
    model_emissions = tracker.final_emissions_data.emissions

    # Evaluate model performance
    if not isinstance(y_test_encoded, (pd.Series, np.ndarray)):
        raise TypeError("y_test should be a pandas Series or numpy array")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("y_pred should be a pandas Series or numpy array")
    evaluate(model_name, y_test_encoded, y_pred, model_emissions)

    # Run comparisons
    run_comparisons()

    # Confusion matrix of the model
    confusion_matrix_plot(y_test_encoded, y_pred)

if __name__ == "__main__":
    main()

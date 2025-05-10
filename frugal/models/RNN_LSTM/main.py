import os
import pandas as pd
import numpy as np
from codecarbon import EmissionsTracker
import gensim.downloader as api
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from frugal.data_preproc.data import load_and_split_data, encoding_and_preproc
from frugal.evaluation import evaluate, run_comparisons, confusion_matrix_plot
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

tracker = EmissionsTracker(log_level="error")

def preprocess_texts(X_train, X_test):
    # Calculate sequence lengths
    train_seq_lengths = [len(seq) for seq in X_train['quote']]
    test_seq_lengths = [len(seq) for seq in X_test['quote']]

    # Determine max_len based on the maximum sequence length
    max_len = max(train_seq_lengths + test_seq_lengths)

    # Load pretrained word2vec model
    w2v_model = api.load('glove-wiki-gigaword-50')

    # Apply embedding
    X_train_embedded = embed_sentences(w2v_model, X_train['quote'])
    X_test_embedded = embed_sentences(w2v_model, X_test['quote'])

    # Padding sequences
    X_train_pad = pad_sequences(X_train_embedded, maxlen=max_len, dtype='float32', padding='post')
    X_test_pad = pad_sequences(X_test_embedded, maxlen=max_len, dtype='float32', padding='post')

    return X_train_pad, X_test_pad, max_len


# Embedding functions
def embed_sentence(model, sentence):
    embedded_sentence = [model.get_vector(word) for word in sentence if word in model.key_to_index]
    return np.array(embedded_sentence)

def embed_sentences(model, sentences):
    return [embed_sentence(model, sentence) for sentence in sentences]


def RNN_LSTM_train(X_train_pad, y_train_encoded, max_len):
    # Parameters
    embedding_dim = 50  # matches the glove-wiki-gigaword-50 dimensions

    # Define the LSTM model
    model = Sequential([
        layers.Masking(mask_value=0., input_shape=(max_len, embedding_dim)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(8, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    # Early stopping
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)

    # Train the model
    model.fit(
        X_train_pad,
        y_train_encoded,
        batch_size=32,
        epochs=4,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    # -- Save the trained model
    model_filename = "RNN_LSTM_train.keras"
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    return model


def main():
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the name of the folder (model name)
    model_name = os.path.basename(script_directory)
    print(model_name)

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Train and predict with model further defined below.
    # This is what is measured for emissions.
    tracker.start()

    # Preprocess the data (embedding and padding)
    X_train_preproc, X_test_preproc, y_train_encoded, y_test_encoded = encoding_and_preproc(X_train, X_test, y_train, y_test)
    X_train_pad, X_test_pad, max_len = preprocess_texts(X_train_preproc, X_test_preproc)

    # Train model (or load existing trained and saved model)
    model = RNN_LSTM_train(X_train_pad, y_train_encoded, max_len)
    #model = load_model("RNN_LSTM_train.keras")

    # Predictions and reports
    y_pred_probs  = model.predict(X_test_pad)
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

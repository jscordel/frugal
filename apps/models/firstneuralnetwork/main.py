from frugal.data_preproc.data import load_and_split_data, encoding_and_preproc
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


def preprocess_texts(X_train, X_test):
    """Tokenize and pad text data."""
    print("initializing")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train["quote"])
    print("fitted")
    print()
    print("type tokenizer")
    print(type(tokenizer))
    print()
    print('-'*15)

    X_train_token = tokenizer.texts_to_sequences(X_train["quote"])
    X_test_token = tokenizer.texts_to_sequences(X_test["quote"])
    print("type X_train_token")
    print(type(X_train_token))
    print()
    print("length X_train_token")
    print(len(X_train_token))
    print()
    print("type X_test_token")
    print(type(X_test_token))
    print()
    print("length X_test_token")
    print(len(X_test_token))
    print()

    vocab_size = len(tokenizer.word_index)
    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='pre')
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='pre')

    print("len(X_train_pad)")
    print(X_train_pad.shape)
    print()
    print("len(X_test_pad)")
    print(X_test_pad.shape)
    print()

    return X_train_pad, X_test_pad, vocab_size

def modelRNN(X_train_pad, y_train, vocab_size):
    """Define, compile, and train the RNN model."""
    embedding_dim = 20
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, mask_zero=True),
        LSTM(10),
        Dense(64, activation='relu'),
        Dense(8, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    model.fit(X_train_pad, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    return model


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    X_train_preproc, X_test_preproc, y_train_encoded, y_test_encoded = encoding_and_preproc(X_train, X_test, y_train, y_test)
    print('-'*15)
    print("len(X_train_preproc)")
    print(X_train_preproc.shape)
    print()
    print("len(X_test_preproc)")
    print(X_test_preproc.shape)
    print()
    print("len(y_train_encoded)")
    print(y_train_encoded.shape)
    print()
    print("len(y_test_encoded)")
    print(y_test_encoded.shape)
    print()
    print('-'*15)

    X_train_pad, X_test_pad, vocab_size = preprocess_texts(X_train_preproc, X_test_preproc)
    model = modelRNN(X_train_pad, y_train_encoded, vocab_size)

    print("X_test_pad shape:", X_test_pad.shape)
    print("y_test_encoded shape:", y_test_encoded.shape)
    print("First few X_test_pad samples:", X_test_pad[:5])
    print("First few y_test_encoded samples:", y_test_encoded[:5])

    model.evaluate(X_test_pad, y_test_encoded)

if __name__ == "__main__":
    main()

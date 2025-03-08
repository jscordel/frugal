from frugal.data_preproc.data import load_and_split_data, encoding_and_preproc
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import gensim.downloader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_split_data()
X_train_preproc, X_test_preproc, y_train_encoded, y_test_encoded = encoding_and_preproc(X_train, X_test, y_train, y_test)

# Calculate sequence lengths
train_seq_lengths = [len(seq) for seq in X_train_preproc['quote']]
test_seq_lengths = [len(seq) for seq in X_test_preproc['quote']]

# Determine max_len based on the maximum sequence length
max_len = max(train_seq_lengths + test_seq_lengths)
print(f'Chosen max_len: {max_len}')

# Parameters
embedding_dim = 50  # matches the glove-wiki-gigaword-50 dimensions

# Load pretrained word2vec model
w2v_model = gensim.downloader.load('glove-wiki-gigaword-50')

# Embedding functions
def embed_sentence(model, sentence):
    embedded_sentence = [model.get_vector(word) for word in sentence if word in model.key_to_index]
    return np.array(embedded_sentence)

def embed_sentences(model, sentences):
    return [embed_sentence(model, sentence) for sentence in sentences]

# Apply embedding
X_train_embedded = embed_sentences(w2v_model, X_train_preproc['quote'])
X_test_embedded = embed_sentences(w2v_model, X_test_preproc['quote'])

# Padding sequences
X_train_pad = pad_sequences(X_train_embedded, maxlen=max_len, dtype='float32', padding='post')
X_test_pad = pad_sequences(X_test_embedded, maxlen=max_len, dtype='float32', padding='post')

# # LSTM PARAMS
# lstm1 = 64
# lstm2 = 32

# Define the LSTM model
model = Sequential([
    layers.Masking(mask_value=0., input_shape=(max_len, embedding_dim)),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dropout(0.3),
    layers.Dense(len(np.unique(y_train_encoded)), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

# Early stopping
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_pad,
    y_train_encoded,
    batch_size=32,
    epochs=1,
    validation_split=0.1,
    callbacks=[early_stop]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test_encoded)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Predictions and reports
predictions = model.predict(X_test_pad)
predicted_classes = np.argmax(predictions, axis=1)
classification_rep = classification_report(y_test_encoded, predicted_classes, output_dict=True)

# Get current folder name
model_folder_name = str(Path(__file__).parent).split("/")[-1]

# Save results to JSON
results = {
    'model_name': model_folder_name,
    'test_loss': loss,
    'test_accuracy': accuracy,
    'classification_report': classification_rep,
    # 'parameters': {
    #     "layer_1_model": "Masking",
    #     "layer_2": f"LSTM {lstm1}"
    # }
}

# # CONFUSION MATRIX
# cm = confusion_matrix(y_test, predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm) # display_labels=ordinal_values
# disp.plot()
# plt.show()

json_file_path = f'/home/mvincendeau/code/jscordel/frugal/apps/evaluations/{model_folder_name}.json'

with open(json_file_path, 'w') as f:
    json.dump(results, f, indent=4)

# Model summary
model.summary()

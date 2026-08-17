import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, GRU, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

# Parameters
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = "bilstm_bigru_model.h5"
TOKENIZER_PATH = "tokenizer.json"

# Load CSV dataset
file_path = "final_dataset.csv"
df = pd.read_csv(file_path)

# Extract texts and labels
texts = df["headline"].astype(str).tolist()
labels = df["label"].values

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Save the tokenizer
with open(TOKENIZER_PATH, 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Split the data into training and testing sets
train_ratio = 0.8
train_size = int(len(data) * train_ratio)
x_train, x_test = data[:train_size], data[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

# Build the BiLSTM + BiGRU Model
input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
embedding_layer = Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(input_layer)
bilstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
bigru_layer = Bidirectional(GRU(64, return_sequences=True))(bilstm_layer)
global_pooling = GlobalMaxPooling1D()(bigru_layer)
dropout = Dropout(0.5)(global_pooling)
output_layer = Dense(1, activation='sigmoid')(dropout)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save the model
model.save(MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")

"""
Author Name: Keketso Justice Tolo
Student Number: 202100092
Course Code: Expert Systems, CS4434

"""

import numpy as np
import tensorflow as tf
import pickle
import os
import json
import sys

# Ensure terminal can print all characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def train_model():
    # Read the text file
    print("Loading data...")
    with open('friends1.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # Tokenize the text
    print("Tokenizing text...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences using n-grams
    print("Creating input sequences...")
    input_sequences = []
    for line in text.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences
    max_sequence_len = max([len(seq) for seq in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Create predictors and label
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # Build the model
    print("Building model...")
    model = Sequential()
    model.add(Embedding(total_words, 100))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(None, max_sequence_len - 1))
    print(model.summary())

    # Train the model
    print("Training model... This may take a while.")
    # Reduced epochs for faster training
    model.fit(X, y, epochs=20, verbose=1)

    # Save the entire model
    print("Saving model and configuration...")
    model.save('model.h5')
    
    # Save model configuration
    model_config = {
        'total_words': total_words,
        'max_sequence_len': max_sequence_len
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(model_config, f)
    
    # Save the tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Training complete. Model, configuration, and tokenizer saved.")

if __name__ == "__main__":
    train_model()
"""
Author Name: Keketso Justice Tolo
Student Number: 202100092
Course Code: Expert Systems, CS4434
"""

import numpy as np
import tensorflow as tf
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def load_resources():
    """
    Load the model, configuration, and tokenizer
    """
    # Load the model
    model = load_model('model.h5')
    
    # Load model configuration for parameters
    with open('model_config.json', 'r') as f:
        model_config = json.load(f)
    
    max_sequence_len = model_config['max_sequence_len']
    
    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return model, tokenizer, max_sequence_len

def predict_next_words(seed_text, next_words=3):
    """
    Predict the next n words based on the seed text
    
    Args:
        seed_text (str): The starting text
        next_words (int): Number of words to predict
        
    Returns:
        str: The seed text plus the predicted words
    """
    model, tokenizer, max_sequence_len = load_resources()
    
    for _ in range(next_words):
        # Tokenize the current seed text
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # Pad the token list
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predict the next word
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        
        # Find the word that corresponds to the predicted index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        # Add the predicted word to the seed text
        seed_text += " " + output_word
    
    return seed_text

if __name__ == "__main__":
    # Example usage
    seed_text = "I will leave if they"
    next_words = 3
    predicted_text = predict_next_words(seed_text, next_words)
    print(f"Seed text: {seed_text}")
    print(f"Predicted text: {predicted_text}")
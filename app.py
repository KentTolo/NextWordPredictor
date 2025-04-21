"""
Author Name: Keketso Justice Tolo
Student Number: 202100092
Course Code: Expert Systems, CS4434
"""
from flask import Flask, render_template, request, jsonify
from predict import predict_next_words
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    seed_text = request.form['seed_text']
    next_words = int(request.form['next_words'])
    
    # Predict the next words
    try:
        predicted_text = predict_next_words(seed_text, next_words)
        return jsonify({'status': 'success', 'predicted_text': predicted_text})
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during prediction: {str(e)}")
        print(error_details)
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Check if required files exist
    required_files = ['model.h5', 'model_config.json', 'tokenizer.pickle']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing required files: {', '.join(missing_files)}")
        print("Please run train_model.py first to generate these files.")
        exit(1)
    
    # Make sure the templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' directory")
    
    # Check if the template file exists
    if not os.path.exists('templates/index.html'):
        print("WARNING: 'templates/index.html' not found. Please make sure to create this file.")
    
    print("Starting web server. Navigate to http://127.0.0.1:5000/ in your browser.")
    app.run(debug=True)
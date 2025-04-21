CS4434 Machine Problem 2 Submission
Student Number: 202100092

Project Description

This submission implements a Next Word Prediction Model using a dataset from friends1.txt. The model is trained using an LSTM-based neural network, and a Flask web interface allows users to input a seed phrase and the number of words to predict. The interface displays the seed text concatenated with the predicted words.

Files Included

**train_model.py: Trains the LSTM model using friends1.txt and saves the model (model.h5), tokenizer (tokenizer.pickle), and configuration (model_config.json).

**app.py: Flask application to serve the web interface.

**predict.py: Contains the prediction logic for generating the next n words.

**templates/index.html: HTML file for the web user interface.

**friends1.txt: Dataset used for training.

**model.h5: Trained model file (generated after running train_model.py).

**tokenizer.pickle: Tokenizer file (generated after running train_model.py).

**model_config.json: Model configuration file.

**ReadMe.txt: This file with instructions.

Prerequisites

Python 3.6 or higher
Required Python packages:
tensorflow (for model training and prediction)
numpy (for numerical operations)
flask (for the web interface)
pickle (for tokenizer serialization)
json (for configuration file handling)

Install the required packages using:

pip install tensorflow numpy flask


How to Run:

Ensure Prerequisites: Verify that Python 3.6 or higher is installed and install the required packages as listed above.

Train the Model:

Run the train_model.py script to train the LSTM model and generate the necessary files (model.h5, tokenizer.pickle, model_config.json):

python train_model.py

This step processes the friends1.txt dataset, trains the model, and saves the model and tokenizer. (You can skip this section as these already provided)



Run the Flask Application:

Ensure the templates directory contains index.html.

Run the app.py script to start the Flask web server:
python app.py

The server will start at http://127.0.0.1:5000/. Open this URL in a web browser to access the interface.


Use the Web Interface:

Enter a seed phrase (e.g., "How you doing") in the text area.
Specify the number of words to predict (1 to 6).
Click "Predict Next Words" to see the predicted text.


Troubleshooting:

If model.h5, tokenizer.pickle, or model_config.json are missing, ensure train_model.py has been run successfully.
If the web interface does not load, verify that index.html is in the templates directory.
Check the console for error messages if predictions fail.




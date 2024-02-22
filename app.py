# Import necessary libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create a Flask web application instance
app = Flask(__name__)

# Load the pre-trained machine learning model using Pickle
model = pickle.load(open('model.pkl', 'rb'))

# Define the route for the home page, which renders the index.html template
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions based on form data
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get input features from the form and convert to integers
    features = [int(x) for x in request.form.values()]

    # Convert the input features to a NumPy array
    final_features = [np.array(features)]

    # Make a prediction using the loaded machine learning model
    prediction = model.predict(final_features)

    # Round the prediction to one decimal place
    output = round(prediction[0], 1)

    # Render the index.html template with the prediction result
    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

# Run the Flask web application in debug mode
if __name__ == "__main__":
    app.run(debug=True)

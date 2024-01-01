import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]  # Get input features from form
    final_features = [np.array(features)] # Convert to numpy array
    prediction = model.predict(final_features) # Make prediction using the model

    output = round(prediction[0], 1) # Round the prediction to one decimal place

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
app = Flask(__name__)

# Function to load the Keras model
def load_keras_model():
    # Clear the session and optimizer state to avoid M1/M2 Mac optimizer warning
    tf.keras.backend.clear_session()
    model = load_model('Final_model_dl.h5')
    return model

# Load your Keras model here
model = load_keras_model()

# Load the fitted scaler from file
with open('fitted_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Route to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Convert the input data into a NumPy array
    input_data = np.array(data).reshape(1, -1)
    
    # Transform the input data using the trained scaler
    input_scaled = scaler.transform(input_data)
    
    # Make predictions on the input data using the loaded model
    prediction = (model.predict(input_scaled) > 0.5).astype(int).flatten()
    
    # Return the prediction as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

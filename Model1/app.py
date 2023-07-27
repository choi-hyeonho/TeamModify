from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

def map_sex_to_int(sex_data):
    sex_mapping = {
        'M': 1,
        'F': 2,
        'I': 3
    }
    try:
        return sex_mapping[sex_data.upper()]
    except KeyError:
        return int(sex_data)

def predict_abalone_age(input_data):
    model_filename = 'abalone_model.pkl'
    with open(model_filename, 'rb') as file:
        model_data = joblib.load(file)
        weight = model_data['weight']
        bias = model_data['bias']

    # Preprocess the input data
    sex_f = map_sex_to_int(input_data['sex'])
    sex_m = map_sex_to_int(input_data['sex'])
    sex_i = map_sex_to_int(input_data['sex'])
    length = float(input_data['length'])
    diameter = float(input_data['diameter'])
    height = float(input_data['height'])
    whole_weight = float(input_data['whole_weight'])
    shucked_weight = float(input_data['shucked_weight'])
    viscera_weight = float(input_data['viscera_weight'])
    shell_weight = float(input_data['shell_weight'])

    # Pass the input data to the model for prediction
    input_vector = np.array([[sex_f, sex_m, sex_i, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]])
    output = np.matmul(input_vector, weight) + bias

    return output[0, 0]


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get user input data
        input_data = request.form

        # Calculate the prediction
        prediction = predict_abalone_age(input_data)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
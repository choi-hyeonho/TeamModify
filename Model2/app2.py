import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle

# 터미널 상에서 input 받아서 예측값 반환 연습 코드

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

# Get input data from the user through the terminal
print("데이터 features 에 대한 8가지 수치들을 space로 구분하여 입력하여주세요: ")
input_str = input()
input_data = np.array(input_str.split(), dtype=float).reshape(1, -1)

# Transform the input data using the trained scaler
input_scaled = scaler.transform(input_data)

# Make predictions on the input data using the loaded model
prediction = (model.predict(input_scaled) > 0.5).astype(int).flatten()

# Print the prediction result to the terminal
print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("축하합니다 맥동성을 찾으셨습니다.")
else:
    print("죄송하지만 아닌 것 같군요(보장은 못함)")

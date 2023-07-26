# app.py
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('binary_classification_data.csv')
X = df.drop('target_class', axis=1)
y = df['target_class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the trained scaler
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler to a file
with open('fitted_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# # Function to load the Keras model
# def load_keras_model():
#     # Clear the session and optimizer state <-- keras optimizer M1/M2 맥에 대한 속도 경고 지우기
#     K.clear_session()
#     model = load_model('Final_model_dl.h5')
#     return model

# # Load your Keras model here
# model = load_keras_model()

# # # Number of samples
# # num_samples = 100

# # # Number of features
# # num_features = 8

# # # Generate random sample data with values between 0 and 1
# # sample_data = np.random.rand(num_samples, num_features)

# # Make predictions on the test data using the loaded model
# y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()

# # Evaluate the model
# accuracy_dl = accuracy_score(y_test, y_pred)
# print("Deep Learning Accuracy:", accuracy_dl)

# # Print classification report
# print("Deep Learning Classification Report:")
# print(classification_report(y_test, y_pred))

# # Print confusion matrix
# conf_matrix_dl = confusion_matrix(y_test, y_pred)
# print("Deep Learning Confusion Matrix:")
# print(conf_matrix_dl)

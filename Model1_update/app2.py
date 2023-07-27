# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib

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
    # 저장된 모델 로드
    model = joblib.load('abalone_model.pkl')

    # 입력 데이터 전처리
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

    # 모델에 입력 데이터 전달하여 결과 예측
    input_vector = np.array([[sex_f, sex_m, sex_i, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]])
    prediction = model.predict(input_vector)

    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # 사용자로부터 입력 데이터 받기
        input_data = request.form

        # 예측 결과 계산
        prediction = predict_abalone_age(input_data)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

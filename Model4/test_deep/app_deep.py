from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# 모델 로드
model = load_model('Model4.h5')

# 라벨 인코더 로드
label_encoder = LabelEncoder()
label_encoder.classes_ = pd.Index(['Bumps', 'Dirtiness', 'K_Scatch', 'Other_Faults', 'Pastry', 'Stains', 'Z_Scratch'])

# Robust Scaling을 위한 함수 정의
def robust_scaling(column):
    median = column.median()
    iqr = column.quantile(0.75) - column.quantile(0.25)
    return (column - median) / iqr

# 강판 불량 입력 데이터를 처리하는 함수
def process_input_data_steel(request):
    # Update: Remove 'Type_of_Steel' from the input_columns list
    input_columns = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
                     'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
                     'Maximum_of_Luminosity', 'Length_of_Conveyer', 'Type_of_Steel_A300', 'Type_of_Steel_A400',
                     'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', #'Outside_X_Index',
                     'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index',
                     'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']



    # 입력값을 데이터에 맞게 변환
    individual_data = []
    for col in input_columns:
        try:
            # 슬라이더 값이 있는 경우 슬라이더 값 사용
            individual_data.append(float(request.form[col + 'Slider']))
        except KeyError:
            # 슬라이더 값이 없는 경우 기본값 0.5 사용
            individual_data.append(0.5)

    # Type_of_Steel 처리
    Type_of_Steel = request.form.get('Type_of_Steel')
    if not Type_of_Steel:  # 값이 선택되지 않은 경우 기본값 0으로 처리
        Type_of_Steel = 0
    else:
        Type_of_Steel = 1

    individual_data.append(Type_of_Steel)  # Type_of_Steel 데이터 추가

    # 개별 데이터를 NumPy 배열로 표현 (target 열 제외)
    data_array = np.array(individual_data, dtype=float).reshape(1, -1)

    return data_array



# Steel 페이지에서 입력 데이터를 처리하는 함수
def process_steel_input():
    data_array = process_input_data_steel(request)

    # 예측 수행
    prediction1 = model.predict(data_array)

    # 타겟 리스트
    target_list = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    # 결과 반환
    num = np.argmax(prediction1)  # NumPy 배열에서 가장 큰 값의 인덱스를 가져옴
    predicted_class = label_encoder.classes_[num]  # 해당 인덱스에 대응하는 클래스 값을 가져옴
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        model_name = request.form['model_name']
        return redirect(url_for('model_page', model_name=model_name))
    return render_template('index.html')

@app.route('/model/<model_name>', methods=['GET', 'POST'])
def model_page(model_name):
    global age_prediction_model, crack_classification_model, mactility_discrimination_model

    if model_name == 'age_prediction':
        model = age_prediction_model
        model_label = 'Abalone Age Prediction Model'
    elif model_name == 'crack_classification':
        model = crack_classification_model
        model_label = 'Steel Plate Crack Classification Model'
    elif model_name == 'mactility_discrimination':
        model = mactility_discrimination_model
        model_label = 'MacDongSeong Discrimination Model'
    else:
        return "Invalid model name."

    if request.method == 'POST':
        # 여기서 해당 모델의 데이터 입력 페이지로 이동하도록 설정
        return redirect(url_for(f'{model_name}_page'))

    # 데이터 입력 페이지를 구성하는 템플릿 파일을 렌더링합니다.
    return render_template('model_page.html', model_label=model_label, model_name=model_name, model=model)

@app.route('/steel_page', methods=['GET', 'POST'])
def steel_page():
    if request.method == 'POST':
        return process_steel_input()
    return render_template('Steel.html')

@app.route('/predict', methods=['POST'])
def predict():
    # '/predict' 엔드포인트에서는 예측을 담당하는 process_steel_input() 함수를 호출하여 결과를 반환
    return process_steel_input()

# Abalone 페이지를 렌더링하는 함수
@app.route('/abalone', methods=['GET'])
def abalone_page():
    return render_template('Abalone_page.html')

# Pulsation 페이지를 렌더링하는 함수
@app.route('/pulsation', methods=['GET'])
def pulsation_page():
    return render_template('Pulsation_page.html')

# index 페이지를 렌더링하는 함수
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

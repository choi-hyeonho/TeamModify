from flask import Flask, render_template, request, redirect, url_for, jsonify
import joblib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import json

app = Flask(__name__)



# ============================================ 스케일러 & 모델 로드

# abalone


# pulsar star
pulsar_scaler_path = os.path.join(os.path.dirname(__file__), 'models/pulsar_scaler.pkl')
pulsar_scaler = joblib.load(pulsar_scaler_path)

def load_keras_model():
    tf.keras.backend.clear_session() # Clear the session and optimizer state to avoid M1/M2 Mac optimizer warning
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models/pulsar_model.h5')
    model = load_model(model_path)
    return model
pulsar_model = load_keras_model()


# steel plate
model_path1 = os.path.join(os.path.dirname(__file__), 'models/steelplate_model1.pkl')
model1 = joblib.load(model_path1)

model_path2 = os.path.join(os.path.dirname(__file__), 'models/steelplate_model2.pkl')
model2 = joblib.load(model_path2)



# ============================================ 입력값 받아와 예측 후 반환 함수

## pulsar star

# 입력값 받아오기
def process_input_data_pulsar(request):
    
    # key에 매치하는 정보 불러오기
    Mean_profile = request.form['MeanProfileSlider']
    Standard_profile = request.form['StandardProfileSlider']
    Excess_profile = request.form['ExcessProfileSlider']
    Skewness_profile = request.form['SkewnessProfileSlider']
    Mean_DM = request.form['MeanDMSlider']
    Standard_DM = request.form['StandardDMSlider']
    Excess_DM = request.form['ExcessDMSlider']
    Skewness_DM = request.form['SkewnessDMSlider']

    # 개별 데이터를 NumPy 배열로 표현
    individual_data = np.array([Mean_profile, Standard_profile, Excess_profile, Skewness_profile,
                                Mean_DM, Standard_DM, Excess_DM, Skewness_DM], dtype=float)
    data_array = individual_data.reshape(1, -1)  # 개별 데이터를 2차원 배열로 변환

    data_array = pulsar_scaler.transform(data_array) # scaling

    return data_array

# 예측값 반환
def process_pulsar_input():
    data_array = process_input_data_pulsar(request)

    # 예측 수행
    prediction = (pulsar_model.predict(data_array) > 0.5).astype(int).flatten()

    if prediction[0] == 1:
        return "축하합니다! 맥동성을 찾으셨습니다"
    else:
        return "아쉽게도 맥동성이 아닙니다"



# steel plate

# 강판 불량 입력 데이터를 처리하는 함수
def process_input_data_steel(request):
    # Update: Change the key names to match the input field names
    X_Minimum = request.form['X_MinimumSlider']
    X_Maximum = request.form['X_MaximumSlider']
    Y_Minimum = request.form['Y_MinimumSlider']
    Y_Maximum = request.form['Y_MaximumSlider']
    Pixels_Areas = request.form['Pixels_AreasSlider']
    X_Perimeter = request.form['X_PerimeterSlider']
    Y_Perimeter = request.form['Y_PerimeterSlider']
    Sum_of_Luminosity = request.form['Sum_of_LuminositySlider']
    Minimum_of_Luminosity = request.form['Minimum_of_LuminositySlider']
    Maximum_of_Luminosity = request.form['Maximum_of_LuminositySlider']
    Length_of_Conveyer = request.form['Length_of_ConveyerSlider']
    Steel_Plate_Thickness = request.form['Steel_Plate_ThicknessSlider']
    Empty_Index = request.form['Empty_IndexSlider']
    Type_of_Steel = request.form['Type_of_Steel']

    # 입력값을 데이터에 맞게 변환
    if Type_of_Steel == 'A300':
        Type_of_Steel = 0
    else:
        Type_of_Steel = 1

    # 개별 데이터를 NumPy 배열로 표현
    individual_data = np.array([X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas,
                                X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity,
                                Maximum_of_Luminosity, Length_of_Conveyer, Steel_Plate_Thickness,
                                Empty_Index, Type_of_Steel], dtype=object)

    data_array = individual_data.reshape(1, -1)  # 개별 데이터를 2차원 배열로 변환
    data_array = np.array(data_array).astype(float)  # str 형태의 데이터를 float으로 변환
    return data_array


# Steel 페이지에서 입력 데이터를 처리하는 함수
def process_steel_input():
    data_array = process_input_data_steel(request)

    # 예측 수행
    prediction1 = model1.predict(data_array)

    # 타겟 리스트
    target_list = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    # 결과 반환
    if prediction1.tolist()[0] == 5 or prediction1.tolist()[0] == 6:
        data_array2 = np.delete(data_array, -2, axis=1)  # 'Empty Index' 데이터 삭제
        prediction2 = model2.predict(data_array2)
        num = prediction2.tolist()[0]
        return target_list[num]
    else:
        num = prediction1.tolist()[0]
        return target_list[num]



# ============================================ 페이지 구현

# home
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

'''
# home
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        model_name = request.form['model_name']
        return redirect(url_for('model_page', model_name=model_name))
    return render_template('index.html')
'''

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



@app.route('/abalone_page', methods=['GET', 'POST'])
def abalone_page():
    global age_prediction_model

    if request.method == 'POST':
        # 모델에 입력 데이터 전달하여 예측
        # 예측 결과를 원하는 방식으로 가공하거나 다른 작업 수행
        # ...

        # 예측 결과를 화면에 출력하거나 다른 동작을 수행하는 코드 추가
        # ...

        return render_template('abalone.html', model=age_prediction_model)

    return render_template('abalone.html', model=age_prediction_model)





# pulsar star
@app.route('/pulsar_page', methods=['GET', 'POST'])
def pulsar_page():
    if request.method == 'POST':
        return process_pulsar_input()
    return render_template('pulsar.html')

@app.route('/pulsar_predict', methods=['POST'])
def pulsar_predict():
    return process_pulsar_input()


# steel plate
@app.route('/steel_page', methods=['GET', 'POST'])
def steel_page():
    if request.method == 'POST':
        return process_steel_input()
    return render_template('steel.html')

@app.route('/steel_predict', methods=['POST'])
def steel_predict():
    return process_steel_input()



if __name__ == '__main__':
    app.run(debug=True)
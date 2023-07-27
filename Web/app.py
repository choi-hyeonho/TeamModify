# Library
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


# ============================================ load scaler & model

# abalone
abalone_path = os.path.join(os.path.dirname(__file__), 'models/abalone_model.pkl')
abalone_model = joblib.load(abalone_path)


# pulsar star
pulsar_scaler_path = os.path.join(os.path.dirname(__file__), 'models/pulsar_scaler.pkl')
pulsar_scaler = joblib.load(pulsar_scaler_path)

def load_pulsar_model():
    tf.keras.backend.clear_session() # Clear the session and optimizer state to avoid M1/M2 Mac optimizer warning
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models/pulsar_model.h5')
    model = load_model(model_path)
    return model
pulsar_model = load_pulsar_model()


# steel plate

# load DL label encoder & model
def load_steel_model():
    tf.keras.backend.clear_session() # Clear the session and optimizer state to avoid M1/M2 Mac optimizer warning
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models/steel_dl_model.h5')
    model = load_model(model_path)
    return model
steel_dl_model = load_steel_model()

label_encoder = LabelEncoder() # encoder
label_encoder.classes_ = pd.Index(['Bumps', 'Dirtiness', 'K_Scatch', 'Other_Faults', 'Pastry', 'Stains', 'Z_Scratch'])


# load ML label encoder & model
model_path1 = os.path.join(os.path.dirname(__file__), 'models/steelplate_model1.pkl')
model1 = joblib.load(model_path1)

model_path2 = os.path.join(os.path.dirname(__file__), 'models/steelplate_model2.pkl')
model2 = joblib.load(model_path2)



# ============================================ 함수

#abalone

'''
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
'''

# 데이터 처리
def process_input_data_abalone(request):

    # Preprocess the input data
    sex = request.form['Sex']
    length = request.form['LengthSlider']
    diameter = request.form['DiameterSlider']
    height = request.form['HeightSlider']
    whole_weight = request.form['Whole_WeightSlider']
    shucked_weight = request.form['Shucked_WeightSlider']
    viscera_weight = request.form['Viscera_WeightSlider']
    shell_weight = request.form['Shell_WeightSlider']

    if sex == '0' : # 수컷
        sex_f, sex_m, sex_i = 0, 1, 0
    elif sex == '1' : # 암컷
        sex_f, sex_m, sex_i = 1, 0, 0
    else :
        sex_f, sex_m, sex_i = 0, 0, 1

    input_array_org = [[sex_f, sex_m, sex_i, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight]]
    input_array = [[float(item) for item in inner_list] for inner_list in input_array_org]
    input_vector = np.array(input_array)

    return input_vector



# 예측값 반환
def process_abalone_input():
    weight = abalone_model['weight']
    bias = abalone_model['bias']

    input_vector = process_input_data_abalone(request)

    # 예측 수행
    output = np.matmul(input_vector, weight) + bias

    return str(output[0, 0])


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

# DL 함수

# Robust Scaling을 위한 함수 정의
def robust_scaling(column):
    median = column.median()
    iqr = column.quantile(0.75) - column.quantile(0.25)
    return (column - median) / iqr

# steel_dl 입력 데이터를 처리하는 함수
def process_input_data_steel_dl(request):
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

# Steel_dl 페이지에서 입력 데이터를 처리하는 함수
def process_steel_input_dl():
    data_array = process_input_data_steel_dl(request)

    '''
    # 각 열에 대해 Robust Scaling을 적용
    for i in range(data_array.shape[1]):
        data_array[:, i] = robust_scaling(data_array[:, i])
    '''

    # 예측 수행
    prediction1 = steel_dl_model.predict(data_array)

    # 타겟 리스트
    target_list = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    # 결과 반환
    num = np.argmax(prediction1)  # NumPy 배열에서 가장 큰 값의 인덱스를 가져옴
    predicted_class = label_encoder.classes_[num]  # 해당 인덱스에 대응하는 클래스 값을 가져옴

    return predicted_class



# ML 함수

# steel_ml 입력 데이터 처리
def process_input_data_steel_ml(request):
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


# steel_ml 예측 후 반환
def process_steel_input_ml():
    data_array = process_input_data_steel_ml(request)

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
# abalone
@app.route('/abalone_page', methods=['GET', 'POST'])
def abalone_page():
    prediction = None
    if request.method == 'POST':
        # Get user input data
        input_data = request.form

        # Calculate the prediction
        prediction = predict_abalone_age(input_data)

    return render_template('abalone.html', prediction=prediction)
'''


@app.route('/abalone_page', methods=['GET', 'POST'])
def abalone_page():  
    if request.method == 'POST':
        return process_abalone_input()

    return render_template('abalone.html')

@app.route('/abalone_predict', methods=['POST'])
def abalone_predict():
    return process_abalone_input()



# pulsar star
@app.route('/pulsar_page', methods=['GET', 'POST'])
def pulsar_page():  
    if request.method == 'POST':
        return process_pulsar_input()

    # model_page에 올릴 데이터
    script_path = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_path, 'static', 'binary_classification_data.csv')
    model_data = pd.read_csv(csv_file_path)
    model_data = model_data.drop(['target_class'], axis=1)
    model_data = model_data.head(10)

    return render_template('pulsar.html', data=model_data.to_html(classes='table table-striped'))

@app.route('/pulsar_predict', methods=['POST'])
def pulsar_predict():
    return process_pulsar_input()


# steel plate DL
@app.route('/steel_page_dl', methods=['GET', 'POST'])
def steel_page_dl():
    if request.method == 'POST':
        return process_steel_input_dl()
    return render_template('steel_dl.html')

@app.route('/steel_predict_dl', methods=['POST'])
def steel_predict_dl():
    # '/predict' 엔드포인트에서는 예측을 담당하는 process_steel_input() 함수를 호출하여 결과를 반환
    return process_steel_input_dl()



# steel plate ML
@app.route('/steel_page_ml', methods=['GET', 'POST'])
def steel_page_ml():
    if request.method == 'POST':
        return process_steel_input_ml()
    return render_template('steel_ml.html')

@app.route('/steel_predict_ml', methods=['POST'])
def steel_predict_ml():
    return process_steel_input_ml()


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# 모델 로드
model_path1 = os.path.join(os.path.dirname(__file__), 'steelplate_model1.pkl')
model1 = joblib.load(model_path1)

model_path2 = os.path.join(os.path.dirname(__file__), 'steelplate_model2.pkl')
model2 = joblib.load(model_path2)


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():

    # 입력 데이터 받기
    X_Minimum = request.form['X_Minimum']
    X_Maximum = request.form['X_Maximum']
    Y_Minimum = request.form['Y_Minimum']
    Y_Maximum = request.form['Y_Maximum']
    Pixels_Areas = request.form['Pixels_Areas']
    X_Perimeter = request.form['X_Perimeter']
    Y_Perimeter = request.form['Y_Perimeter']
    Sum_of_Luminosity = request.form['Sum_of_Luminosity']
    Minimum_of_Luminosity = request.form['Minimum_of_Luminosity']
    Maximum_of_Luminosity = request.form['Maximum_of_Luminosity']
    Length_of_Conveyer = request.form['Length_of_Conveyer']
    Steel_Plate_Thickness = request.form['Steel_Plate_Thickness']
    Empty_Index = request.form['Empty_Index']
    Type_of_Steel = request.form['Type_of_Steel']

    # 입력값을 데이터에 맞게 변환
    if Type_of_Steel == 'A300' :
        Type_of_Steel = 0
    else :
        Type_of_Steel = 1

    # 개별 데이터를 NumPy 배열로 표현
    individual_data = np.array([X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas,
         X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity,
         Maximum_of_Luminosity, Length_of_Conveyer, Steel_Plate_Thickness,
         Empty_Index, Type_of_Steel], dtype=object)

    data_array = individual_data.reshape(1, -1) # 개별 데이터를 2차원 배열로 변환
    data_array = np.array(data_array).astype(int) # str 형태의 데이터를 int로 변환

    print('-------------->', data_array)

    # 예측 수행
    prediction1 = model1.predict(data_array)

    # 타겟 리스트
    target_list = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    # 결과 반환
    if prediction1.tolist()[0] == 5 or prediction1.tolist()[0] == 6 :
        data_array2 = np.delete(data_array, -2, axis=1)  # 'Empty Index' 데이터 삭제
        prediction2 = model2.predict(data_array2)
        num = prediction2.tolist()[0]
        return target_list[num]
        
    else :
        num = prediction1.tolist()[0]
        return target_list[num]

if __name__ == '__main__':
    app.run()
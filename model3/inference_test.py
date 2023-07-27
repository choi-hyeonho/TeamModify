import joblib
import os
import numpy as np

# 모델 로드
model_path1 = os.path.join(os.path.dirname(__file__), 'steelplate_model1.pkl')
model1 = joblib.load(model_path1)

model_path2 = os.path.join(os.path.dirname(__file__), 'steelplate_model2.pkl')
model2 = joblib.load(model_path2)

model_path3 = os.path.join(os.path.dirname(__file__), 'steelplate_model3.pkl')
model3 = joblib.load(model_path3)

# 테스트 변수 지정
X_Minimum = 805
X_Maximum = 811
Y_Minimum = 598090
Y_Maximum = 598094
Pixels_Areas = 16
X_Perimeter = 6
Y_Perimeter = 4
Sum_of_Luminosity = 2034
Minimum_of_Luminosity = 110
Maximum_of_Luminosity = 140
Length_of_Conveyer = 1360
Steel_Plate_Thickness = 50
Empty_Index = 0.3333
Type_of_Steel = 1

# 개별 데이터를 NumPy 배열로 표현
individual_data = np.array([X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas,
         X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity,
         Maximum_of_Luminosity, Length_of_Conveyer, Steel_Plate_Thickness,
         Empty_Index, Type_of_Steel])

# 개별 데이터를 2차원 배열로 변환
data_array = individual_data.reshape(1, -1)

# 1차 예측 수행
prediction1 = model1.predict(data_array)

# 타겟 리스트
target_list = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

if prediction1.tolist()[0] == 0 or prediction1.tolist()[0] == 5 or prediction1.tolist()[0] == 6 : # 예측값이 0, 5, 6인 경우
    
    # 2차 예측 수행
    data_array2 = np.delete(data_array, -2, axis=1)  # 'Empty Index' 데이터 삭제
    prediction2 = model2.predict(data_array2) 

    if prediction1.tolist()[1] == 0 : # 2차 예측시 1인 경우

        # 3차 예측 수행
        prediction3 = model3.predict(data_array2) # data_array2 데이터 그대로 사용
        num = prediction2.tolist()[0]
        print(target_list[num])
              
    else : # 2차 예측시 0인 경우 3차 예측 거치지 않음
        num = prediction2.tolist()[0]
        print(target_list[num])

else : # 1차 예측시 1, 2, 3, 4인 경우 2,3차 예측 거치지 않음
    num = prediction1.tolist()[0]
    print(target_list[num])
import joblib
import os
import numpy as np

# 모델 로드
model_path1 = os.path.join(os.path.dirname(__file__), 'steelplate_model1.pkl')
model1 = joblib.load(model_path1)

model_path2 = os.path.join(os.path.dirname(__file__), 'steelplate_model2.pkl')
model2 = joblib.load(model_path2)

# 테스트 변수 지정
X_Minimum = 77
X_Maximum = 77
Y_Minimum = 77
Y_Maximum = 77
Pixels_Areas = 77
X_Perimeter = 77
Y_Perimeter = 77
Sum_of_Luminosity = 77
Minimum_of_Luminosity = 77
Maximum_of_Luminosity = 77
Length_of_Conveyer = 77
Steel_Plate_Thickness = 77
Empty_Index = 77
Type_of_Steel = 0

# 개별 데이터를 NumPy 배열로 표현
individual_data = np.array([X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas,
         X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity,
         Maximum_of_Luminosity, Length_of_Conveyer, Steel_Plate_Thickness,
         Empty_Index, Type_of_Steel])

# 개별 데이터를 2차원 배열로 변환
data_array = individual_data.reshape(1, -1)

# 예측 수행
prediction1 = model1.predict(data_array)

# 타겟 리스트
target_list = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

# 결과 반환
if prediction1.tolist()[0] == 5 or prediction1.tolist()[0] == 6 :
    data_array2 = np.delete(data_array, -2, axis=1)  # 'Empty Index' 데이터 삭제
    prediction2 = model2.predict(data_array2) # 예측
    num = prediction2.tolist()[0]
    print(target_list[num])
else :
    num = prediction1.tolist()[0]
    print(target_list[num])

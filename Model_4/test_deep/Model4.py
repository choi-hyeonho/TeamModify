from sklearn.utils import class_weight
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
import joblib
import h5py

# CSV 파일 읽기
df = pd.read_csv(r"C:\Users\chh-9\Section6/mulit_classification_data.csv")



# 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults' 컬럼 제외하고 'df2' 생성
df2 = df.drop(columns=['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'])

# 'df' 데이터프레임에서 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults' 컬럼들만 추출
target = df[['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']].copy()

# 빈 리스트를 생성합니다.
target_values = []

# 'target' 데이터프레임의 모든 행에 대해 반복합니다.
for index, row in target.iterrows():
    # 각 행에서 값이 1인 컬럼명을 찾아서 리스트에 추가합니다.
    target_values.append(", ".join([column for column in target.columns if row[column] == 1]))

# 'target'이라는 새로운 열을 추가하여 새로운 데이터프레임을 생성합니다.
new_df = target.assign(target=target_values)

df3 = df2.copy()

# 'target' 데이터프레임에서 'target' 열만 추출합니다.
target_column = new_df['target']

# 'df3' 데이터프레임에 'target' 열을 추가합니다.
df3['target'] = target_column

# 입력 피처와 클래스를 나눕니다.
X = df3.drop(columns=['target'])  # 'target' 열을 제외한 모든 열을 입력 피처로 선택합니다.
y = df3['target']  # 'target' 열을 클래스로 선택합니다.

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df3['target'])

# 클래스와 숫자 매핑을 확인합니다.
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Robust Scaling을 위한 함수 정의
def robust_scaling(column):
    median = column.median()
    iqr = column.quantile(0.75) - column.quantile(0.25)
    return (column - median) / iqr

# clean_df에서 특성들만 선택하여 Robust Scaling 적용
# 여기서는 모든 열에 Robust Scaling을 적용하므로 apply 함수를 사용합니다.
scaled_features = df3.drop(columns=['target']).apply(robust_scaling)

# 데이터프레임에서 X와 y 분리
X = scaled_features
y = y_encoded  # 라벨 인코딩된 형태로 사용

# 데이터를 train-validation-test로 나누기
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# EarlyStopping 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# 학습 이전 시간 기록
start_time = time.time()

# 모델 구성
model = Sequential()
input_shape = (27,)  # 입력 데이터의 형태에 맞게 설정
model.add(Dense(64, activation='relu', input_shape=input_shape))  # 수정된 입력 레이어
model.add(Dense(32, activation='relu'))  # 두 번째 Dense Layer
model.add(Dense(7, activation='softmax'))  # 출력 레이어 (7개의 클래스에 대한 소프트맥스 활성화 함수)

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.00075), metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation), callbacks=[early_stopping])

# 학습 후 시간 기록
end_time = time.time()

# 학습에 소요된 시간 계산
training_time = end_time - start_time

# 학습에 소요된 시간 출력
print("Training time:", training_time, "seconds")

# 훈련 데이터의 성능 지표 계산
y_train_pred = model.predict(X_train)
y_train_pred_class = np.argmax(y_train_pred, axis=1)
f1_train_score = f1_score(y_train, y_train_pred_class, average='weighted')
accuracy_train = accuracy_score(y_train, y_train_pred_class)

# 시험 데이터의 성능 지표 계산
y_test_pred = model.predict(X_test)
y_test_pred_class = np.argmax(y_test_pred, axis=1)
f1_test_score = f1_score(y_test, y_test_pred_class, average='weighted')
accuracy_test = accuracy_score(y_test, y_test_pred_class)

# 훈련 데이터와 시험 데이터의 성능 지표 출력
print("Training F1 Score:", f1_train_score)
print("Training Accuracy:", accuracy_train)
print("Test F1 Score:", f1_test_score)
print("Test Accuracy:", accuracy_test)

# 학습된 모델을 피클링하여 저장
#model_filename = 'trained_model.pkl'
#joblib.dump(model, model_filename)

# 학습된 모델을 HDF5 형식으로 저장
model.save('Model4.h5')
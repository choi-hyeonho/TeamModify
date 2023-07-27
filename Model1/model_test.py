# app.py 파일
import os
import joblib
from model import abalone_exec

# 모델 학습 및 저장
LEARNING_RATE = 0.1
abalone_exec(epoch_count=2000, mb_size=100, report=20)

# 저장된 모델 로드
current_directory = os.getcwd()
model_path = os.path.join(current_directory, 'abalone_model.pkl')
model = joblib.load(model_path)
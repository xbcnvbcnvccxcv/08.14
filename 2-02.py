import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 데이터 로드
header = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('./data/2.iris.csv', names=header)

# 문자열 값을 숫자로 변환
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

# 데이터 준비
array = data.values
X = array[:, 0:4]  # 특성
Y = array[:, 4]    # 타겟

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Shapes: X_test: {X_test.shape}, X_train: {X_train.shape}, Y_test: {Y_test.shape}, Y_train: {Y_train.shape}")

# 모델 생성 및 훈련
model = LinearRegression()
model.fit(X_train, Y_train)

# 모델 예측
y_pred = model.predict(X_test)

# 성능 평가
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual values')
plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted values')
plt.legend()
plt.xlabel("Test Samples")
plt.ylabel("Target Value")
plt.title("Actual vs Predicted Values")
plt.savefig("./results/plot.png")  # 경로와 파일 확장자를 포함하여 그래프 저장
plt.savefig("./WW/")

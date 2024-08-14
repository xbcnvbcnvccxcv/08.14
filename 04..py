#교수님
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv('./data/1.salary.csv')
#데이터 프레임 제거 및 독립/종속변수 설정
array = data.values
X = array[:,0] # [행,열]
Y = array[:,1]
X = X.reshape(-1,1)
# 데이터 분할(Train, Test)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
print((X_test.shape, X_test.shape, Y_train.shape, Y_test.shape))

#모델 선택 및 학습
#방정식 찾게하기
model = LinearRegression()
model.fit(X_train, Y_train)
model.coef_
model.intercept_

#모델 예측
y_pred = model.predict(X_test) #근속연속이 있을 때 연봉을 예측해봐
error = mean_absolute_error(y_pred, Y_test)
print(error)

plt.clf()
plt.scatter(X_test, Y_test, color='blue', label='Actual values')
plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted values', marker='o')
plt.legend()
plt.xlabel("Exoerience Years(Year)")
plt.ylabel("Salary($)")
plt.savefig("./WW/")
# plt.show()
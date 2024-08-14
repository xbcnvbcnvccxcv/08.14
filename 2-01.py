import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header= ['sepal-length','sepal-width','petal-length','petal-width','class']
data = pd.read_csv('./data/2.iris.csv', names = header)

array = data.values
X = array[:,0:4] # [행,열]
Y = array[:,4]
X = X.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
print((X_test.shape, X_test.shape, Y_train.shape, Y_test.shape))

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




y_pred = model.predict(X_test)
confusion_matrix(y_pred, Y_test)
print(confusion_matrix(y_pred, Y_test))
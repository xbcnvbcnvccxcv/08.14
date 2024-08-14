import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import  scipy

header= ['Experience Years','Salary']
data = pd.read_csv('./data/1.salary.csv', names = header)

array = data.values
array.shape
X = array[:, 0]
Y = array[:, 1]

fig = plt.figure()
ax = fig.add_subplot()
plt.clf()

plt.scatter(X,Y, label='random', color='gold', marker='*', s=30, alpha=0.5)
X1 = X.reshape(-1,1)

(X_train, X_text, Y_train, Y_test) = train_test_split(X,Y, test_size = 0.2)
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_text)
print(y_pred)

plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)),Y_test,color='blue')

plt.plot(range(len(y_pred)),y_pred,color='r',marker='x')

plt.show()

mae=mean_absolute_error(y_pred,Y_test)
print(mae)




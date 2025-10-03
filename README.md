# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare and split the dataset.

2.Scale target values and train the SGD Regressor.

3.Predict on test data and compute metrics.

4.Plot actual vs predicted prices.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: R.Sairam
RegisterNumber:  25000694
*/
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

X = np.array([[1200, 3], [1500, 4], [800, 2], [2000, 5], [1700, 4], [1000, 3]])

y = np.array([200000, 250000, 150000, 320000, 280000, 180000])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler_X = StandardScaler()

scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)

X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()

sgd = SGDRegressor(max_iter=2000, tol=1e-3, learning_rate='constant', eta0=0.001, random_state=42)

sgd.fit(X_train_scaled, y_train_scaled)

y_pred_scaled = sgd.predict(X_test_scaled)

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

print("R² Score:", r2_score(y_test, y_pred))

print("\nComparison (Actual vs Predicted):")

for actual, pred in zip(y_test, y_pred):

  print("Actual: ",actual, Predicted: ",pred)

plt.scatter(y_test, y_pred, color='red', label='Predicted vs Actual')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],color='blue', linestyle='--', label='Predicted')

plt.title("SGD Regression - House Prices")

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.legend()

plt.show()


## Output: 
<img src="ex4 output 1.png" alt="Output" width="500">

<img src="ex4 output 2.png" alt="Output" width="500">

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

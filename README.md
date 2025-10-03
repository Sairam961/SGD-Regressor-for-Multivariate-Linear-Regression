# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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

X = np.array([[1200, 3], [1500, 4], [800, 2], [2000, 5], [1700, 4], [1000, 3]])

y = np.array([200000, 250000, 150000, 320000, 280000, 180000])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01)

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

print("R² Score:", r2_score(y_test, y_pred))

print("Price of the house:",y_pred)

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Price')

plt.scatter(range(len(y_test)), y_pred, color='red', marker='x', label='Predicted Price')

plt.title("Actual vs Predicted House Prices")

plt.xlabel("Test House")

plt.ylabel("Price")

plt.legend()

plt.show()




## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

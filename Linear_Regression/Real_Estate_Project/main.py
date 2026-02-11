import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Real_Estate.csv')

X = dataset.iloc[:, 3 : 4].values
y = dataset.iloc[: , -1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1 / 3, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

print('Model Learned')
print(f'Base Price (Intercept) : {regressor.intercept_ : .2f}')
print(f'Price Drop Per Meter (Slope) : {regressor.coef_[0] : .4f}')

plt.figure(figsize = (10,6))
plt.scatter(x_train, y_train, color = 'red', s = 10, label = 'Real Prices')
plt.plot(x_train, regressor.predict(x_train), color = 'blue', linewidth = 2, label = 'Predicted Prices')
plt.title('House Price vs Distance to MRT Station')
plt.xlabel('Distance to MRT (meters)')
plt.ylabel('House Price (Unit Area)')
plt.legend()
plt.show()
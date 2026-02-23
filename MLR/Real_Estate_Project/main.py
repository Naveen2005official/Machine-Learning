import numpy as np  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Real_Estate.csv')

X = dataset.iloc[ : , 2 : 7].values
y = dataset.iloc[ : , 7].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print("Model's Logic")
print(f"Base Price(Intercept) : {regressor.intercept_ : .2f}\n")

print("Multipliers (Coefficients):")
feature_names = ['House Age', 'Distance to MRT', 'Number of Stores', 'Latitude', 'Longitude']
for name, coef in zip(feature_names, regressor.coef_):
    print(f"{name}: {coef:.4f}")

y_pred = regressor.predict(x_test)

print("\n--- Reality vs. Prediction (First 10 Houses) ---")
comparison_df = pd.DataFrame({
    'Actual Price': y_test[:10],
    'Predicted Price': np.round(y_pred[:10], 2),
    'Error (Difference)': np.round(y_test[:10] - y_pred[:10], 2)
})
print(comparison_df)
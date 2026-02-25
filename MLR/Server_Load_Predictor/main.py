import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv("Server_Logs.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

print(f"Model's Logic (Base CPU Load) : {regressor.intercept_:.2f}\n")

print("Multipliers (Coefficients):")
feature_names = ['Active Users', 'Requests Per Second', 'Background Tasks']
for name, coef in zip(feature_names, regressor.coef_):
    print(f"{name} : {coef:.4f}")

y_pred = regressor.predict(x_test)
y_pred = np.clip(y_pred, 1, 99)

print("\n--- Reality vs. Prediction (First 10 Logs) ---\n")
comparison_df = pd.DataFrame({
    'Actual CPU Load': y_test[:10],
    'Predicted CPU Load': np.round(y_pred[:10], 2),
    'Error (Difference)': np.round(y_test[:10] - y_pred[:10], 2)
})
print(comparison_df)

print("\n--- Model Performance ---")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse : .4f}")
print(f"R² Score: {r2 * 100 : .2f}")

plt.figure(figsize = (8, 6))
plt.scatter(y_test, y_pred, color = 'blue', alpha = 0.6, label = 'Predictions')
min_v = min(y_test.min(), y_pred.min())
max_v = max(y_test.max(), y_pred.max())
plt.plot([min_v, max_v], [min_v, max_v], color = 'red', linewidth = 2, label = 'Perfect Prediction (x = y)')
plt.title('Server CPU Load: Actual vs. Predicted')
plt.xlabel('Actual CPU Load (%)')
plt.ylabel('Predicted CPU Load (%)')
plt.legend()
plt.grid(True)
plt.show()
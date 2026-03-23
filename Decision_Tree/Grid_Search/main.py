import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC, export_text as et
from sklearn.metrics import classification_report as cr, confusion_matrix as cm

# 1. LOAD & SPLIT DATA
data = pd.read_csv('Server_Logs.csv') # Assuming your file from the last run!
X = data.iloc[ : , : -1].values
y = data.iloc[ : , -1].values

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

# 2. DEFINE THE GRID (The parameters to test)
# The AI will test every single combination of these settings:
param_grid = {
    'max_depth': [2, 3, 4, 5, 10],            # How deep should the tree go?
    'min_samples_split': [2, 5, 10],          # Minimum servers needed to split a rule
    'criterion': ['gini', 'entropy']          # The math used to calculate confusion
}

# 3. SETUP GRID SEARCH
# cv=5 means "Cross-Validation" (it tests each config 5 times to be sure)
# scoring='recall' tells it to prioritize catching the 1.0 class (Crashes!)
grid_search = GridSearchCV(
    estimator=DTC(random_state=42), 
    param_grid=param_grid, 
    cv=5, 
    scoring='recall' 
)

print("⚙️ Running Grid Search... Testing 30 different Tree configurations...")
grid_search.fit(X_train, y_train)

# 4. EXTRACT THE WINNING MODEL
best_tree = grid_search.best_estimator_

print("\n--- 🏆 Grid Search Complete! ---")
print("The AI found the perfect settings:")
print(grid_search.best_params_)

# 5. TEST THE WINNING MODEL
y_pred = best_tree.predict(X_test)

print("\n--- 🚨 Optimized Diagnostics ---")
print("Confusion Matrix:\n", cm(y_test, y_pred))
print("\nClassification Report:\n", cr(y_test, y_pred))

print("\n--- The Optimized Runbook ---")
print(et(best_tree, feature_names=data.columns[:-1].tolist()))
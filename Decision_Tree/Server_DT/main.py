import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier as DTC, export_text as et, plot_tree as pt
from sklearn.metrics import classification_report as cr, confusion_matrix as cm, accuracy_score as acs

data = pd.read_csv('Server_Logs.csv')
print("Dataset Loaded Successfully!\n")

X = data.iloc[ : , : -1].values
y = data.iloc[ : , -1].values

X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state=42)
print("Data Split into Training and Testing Sets!\n")

classifier = DTC(max_depth=3)
classifier.fit(X_train, y_train)
print("Model Trained Successfully!\n")

y_pred = classifier.predict(X_test)
print("Predictions Made on Test Set!\n")
print("Predicted Values: \n", y_pred)

print("\nClassification Report:\n", cr(y_test, y_pred))
print("Confusion Matrix:\n", cm(y_test, y_pred))
print("Accuracy Score: ", acs(y_test, y_pred))

print("\nDecision Tree Structure:\n", et(classifier, feature_names = data.columns[:-1].tolist()))

plt.figure(figsize=(15,10))
pt(classifier, feature_names = data.columns[:-1].tolist(), class_names = ['Normal', 'Anomalous'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
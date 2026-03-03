import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)
cpu = np.random.randint(10, 100, 200)
ram = np.random.randint(10, 100, 200)
status = np.where(cpu + ram + np.random.normal(0, 10, 200) > 150, 1, 0)

X = np.column_stack((cpu, ram))
y = status

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("--- 📊 Diagnostic Dashboard ---")
print(f"\nOverall Accuracy : {accuracy : .2f}%\n")
print("Confusion Matrix:")
print(f"[{cm[0][0]} (TN)]  [{cm[0][1]} (FP)]")
print(f"[{cm[1][0]} (FN)]  [{cm[1][1]} (TP)]\n")

print("--- 🚨 Breakdown ---")
print(f"True Negatives (Safe, predicted Safe): {cm[0][0]}")
print(f"True Positives (Crashed, predicted Crashed): {cm[1][1]}")
print(f"False Positives (Fake Alarms): {cm[0][1]}")
print(f"False Negatives (Missed Crashes): {cm[1][0]}")

print("\n--- 📈 Performance Metrics ---")
print(f"Precision: {precision : .2f}")
print(f"Recall: {recall : .2f}")
print(f"F1 Score: {f1 : .2f}")
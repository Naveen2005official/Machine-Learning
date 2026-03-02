import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    'CPU_Load': [20, 35, 45, 10, 50,  80, 95, 85, 90, 99],
    'RAM_Usage': [30, 40, 50, 20, 60,  85, 90, 95, 80, 95],
    'Status':    [0,  0,  0,  0,  0,   1,  1,  1,  1,  1] 
}

df = pd.DataFrame(data)

X = df.iloc[ :, : -1].values
y = df.iloc[ :, -1].values

classifier = LogisticRegression()
classifier.fit(X, y)

print("Model Trained Successfully!\n")

new_servers = np.array([
    [30, 40],
    [95, 95],
    [65, 65]
])

predictions = classifier.predict(new_servers)
probabilities = classifier.predict_proba(new_servers)

print(probabilities)

print("\n--- 📊 Diagnostics Report ---")
for i in range(len(new_servers)):
    cpu, ram = new_servers[i]
    pred = predictions[i]
    prob_safe = probabilities[i][0] * 100
    prob_crash = probabilities[i][1] * 100
    
    status = "🔴 CRASHED" if pred == 1 else "🟢 SAFE"
    
    print(f"Server {i+1} [CPU: {cpu}%, RAM: {ram}%] -> {status}")
    print(f"   Model Confidence: {prob_safe:.1f}% Safe | {prob_crash:.1f}% Crash\n")
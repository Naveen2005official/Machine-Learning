import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

data = {
    'CPU_Load': [20, 35, 45, 10, 50,  80, 95, 85, 90, 99],
    'RAM_Usage': [30, 40, 50, 20, 60,  85, 90, 95, 80, 95],
    'Status':    [0,  0,  0,  0,  0,   1,  1,  1,  1,  1] 
}
df = pd.DataFrame(data)

X = df[['CPU_Load', 'RAM_Usage']].values
y = df['Status'].values

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X, y)

print("--- 🌳 Decision Tree Grown! ---\n")

tree_rules = export_text(tree_model, feature_names=['CPU_Load', 'RAM_Usage'])
print("--- The AI's Runbook ---")
print(tree_rules)

plt.figure(figsize=(8, 5))
plot_tree(tree_model, 
          feature_names=['CPU_Load', 'RAM_Usage'], 
          class_names=['Safe', 'Crashed'], 
          filled=True, 
          rounded=True)

plt.title("Server Crash Decision Tree")
plt.show()
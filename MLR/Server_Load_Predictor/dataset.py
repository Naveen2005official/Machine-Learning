import numpy as np
import pandas as pd

np.random.seed(42)

n = 500

users = np.random.randint(100, 2000, n)

rps = users * np.random.uniform(0.5, 2.0, n)

bgt = np.random.randint(10, 100, n)

cpu_load = 5 + (0.01 * users) + (0.02 * rps) + (0.1 * bgt) + np.random.normal(0, 3, n)
cpu_load = np.clip(cpu_load, 1, 99)

df = pd.DataFrame({
    "Active_Users": users,
    "Requests_Per_Second": np.round(rps, 2),    
    "Background_Tasks": bgt,
    "CPU_Load": np.round(cpu_load, 2)
})

df.to_csv("Server_Logs.csv", index=False)
print("Dataset created and saved as 'Server_Logs.csv'")
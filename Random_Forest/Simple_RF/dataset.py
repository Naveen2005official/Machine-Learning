import numpy as np
import pandas as pd

print("Starting Server Log Simulation...")

np.random.seed(42)
n = 100
cpu_load = np.random.randint(10, 100, n)
ram_usage = np.random.randint(10, 100, n)
disk_usage = np.random.randint(10, 100, n)

status = np.zeros(n)
for i in range(n):
    if(cpu_load[i] > 80 and ram_usage[i] > 80) or disk_usage[i] > 95 : 
        status[i] = 1

chaos_indicies = np.random.choice(n, 5, replace=False)
status[chaos_indicies] = 1 - status[chaos_indicies]

data = pd.DataFrame({
    'CPU_Load' : cpu_load,
    'RAM_Usage' : ram_usage,
    'Disk_Usage' : disk_usage,
    'Status' : status
})

data.to_csv('Server_Logs.csv', index = False)
print("Server Log Simulation Completed! Dataset saved as 'Server_Logs.csv'.")
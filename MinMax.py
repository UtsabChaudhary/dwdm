import numpy as np

# Sample data
data = np.array([15, 25, 35, 45, 55])

print("Original Data:")
print(data)

# Calculate minimum and maximum values
min_value = np.min(data)
max_value = np.max(data)

# Apply Min-Max Scaling
scaled_data = (data - min_value) / (max_value - min_value)

print("\nMinimum Value:", min_value)
print("Maximum Value:", max_value)
print("\nMin-Max Scaled Data:")
print(scaled_data)
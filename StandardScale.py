import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create a DataFrame
data = {
    'Marks': [45, 50, 55, 60, 65]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply Standard Scaling
df['Scaled_Marks'] = scaler.fit_transform(df[['Marks']])

print("\nData After Standard Scaling:")
print(df)
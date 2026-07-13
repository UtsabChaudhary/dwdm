import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Create DataFrame
df = pd.DataFrame({
    'Marks': [35, 45, 55, 65, 75]
})

print("Original Data:")
print(df)

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
df['MinMax_Scaled'] = minmax_scaler.fit_transform(df[['Marks']])

# Standard Scaling
standard_scaler = StandardScaler()
df['Standard_Scaled'] = standard_scaler.fit_transform(df[['Marks']])

print("\nData After Scaling:")
print(df)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load CPS dataset
data = pd.read_csv("sample_cps_data.csv")

# Remove missing values
data = data.dropna()

# Select numerical features
features = data.select_dtypes(include=["float64", "int64"])

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(features)

print("Dataset shape:", data.shape)
print("Preprocessing completed.")

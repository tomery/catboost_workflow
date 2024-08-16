# generate_data.py
import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Create data directory
os.makedirs('data', exist_ok=True)

# Save to CSV
df.to_csv('data/data.csv', index=False)

print("Data saved to 'data/data.csv'")

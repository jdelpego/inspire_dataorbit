import pandas as pd

# Load dataset
df = pd.read_csv("realtor-data.csv")
# Display first few rows and column names
print(df.head())
print("\nColumn Names:", df.columns)

import pandas as pd

df = pd.read_csv("data/flood.csv")

print("Shape (rows, columns):", df.shape)
print("\nColumn names:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values per column:\n", df.isna().sum())


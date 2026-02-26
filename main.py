import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/flood.csv")

# Seperate features and target variable

x = df.drop(columns=["FloodProbability"]) # inputs
y= df["FloodProbability"] # output

# Split the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("Training rows:",x_train.shape[0])
print("Testing rows:",x_test.shape[0])



print("Shape (rows, columns):", df.shape)
print("\nColumn names:\n", df.columns)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values per column:\n", df.isna().sum())


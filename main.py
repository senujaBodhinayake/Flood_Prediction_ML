import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/flood.csv")

# Seperate features and target variable

x = df.drop(columns=["FloodProbability"]) # inputs
y_cont= df["FloodProbability"] # output

# Split the data into training and testing sets
x_train,x_test,y_train_cont,y_test_cont = train_test_split(x,y_cont,test_size=0.2,random_state=42)

# Create threshold from train data only
threshold = y_train_cont.mean()

#convert continuous target variable to binary using the threshold
y_train = (y_train_cont >= threshold).astype(int)
y_test = (y_test_cont >= threshold).astype(int)




print("Threshold used:", threshold)
print("Train label counts (Low,High):", y_train.value_counts().sort_index().to_dict())
print("Test label counts (Low,High):", y_test.value_counts().sort_index().to_dict())
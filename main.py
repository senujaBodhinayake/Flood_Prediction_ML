import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/flood.csv")

# Separate features and target variable
x = df.drop(columns=["FloodProbability"]) 
y_cont = df["FloodProbability"]

# Split the data into training and testing sets
x_train, x_test, y_train_cont, y_test_cont = train_test_split(
    x, y_cont, test_size=0.2, random_state=42
)

# Set seed for reproducibility
np.random.seed(42)

# Add small noise to the features to simulate real-world measurement error
x_train_noisy = x_train + np.random.normal(0, 0.05, x_train.shape)
x_test_noisy = x_test + np.random.normal(0, 0.05, x_test.shape)

# Create threshold from train data only
threshold = y_train_cont.median()

# Convert continuous target variable to binary
y_train = (y_train_cont >= threshold).astype(int)
y_test = (y_test_cont >= threshold).astype(int)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(x_train_noisy, y_train)

# Make predictions
y_pred = model.predict(x_test_noisy)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Get probability predictions
y_prob = model.predict_proba(x_test_noisy)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("PR AUC Score:", pr_auc)
print(confusion_matrix(y_test, y_pred))
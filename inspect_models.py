import joblib

print("Loading Linear Regression Model...")
lr = joblib.load('linear_regression_model.joblib')
print("LR Type:", type(lr))
if hasattr(lr, 'n_features_in_'):
    print("LR Features Number:", lr.n_features_in_)
if hasattr(lr, 'feature_names_in_'):
    print("LR Feature Names:", list(lr.feature_names_in_))
else:
    print("LR Model Feature names not found.")

print("\nLoading Random Forest Model...")
rf = joblib.load('random_forest_model.joblib')
print("RF Type:", type(rf))
if hasattr(rf, 'n_features_in_'):
    print("RF Features Number:", rf.n_features_in_)
if hasattr(rf, 'feature_names_in_'):
    print("RF Feature Names:", list(rf.feature_names_in_))
else:
    print("RF Model Feature names not found.")

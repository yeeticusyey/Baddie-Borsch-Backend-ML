import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import json
import os
import joblib  # Add this import to save the column order

# Parse the JSON data from the file (adjust the path to your actual JSON file)
cwd = os.getcwd()
json_file_path = os.path.join(cwd, "data", "Bosch_Dataset.json")

with open(json_file_path, 'r') as file:
    data = json.load(file)

# Convert JSON data to pandas DataFrame
df = pd.DataFrame(data)

# Convert date fields to datetime objects
df['lastCalibration'] = pd.to_datetime(df['lastCalibration'], errors='coerce')
df['calibrationDue'] = pd.to_datetime(df['calibrationDue'], errors='coerce')

# Handle datetime columns by calculating the number of days since the first calibration date
df['lastCalibration_days'] = (df['lastCalibration'] - df['lastCalibration'].min()).dt.days
df['calibrationDue_days'] = (df['calibrationDue'] - df['lastCalibration'].min()).dt.days

# Feature extraction: calculate days until calibration is due
df['daysUntilCalibrationDue'] = (df['calibrationDue'] - df['lastCalibration']).dt.days

# Convert categorical columns into numeric using Label Encoding
label_cols = ['div', 'description', 'standard', 'category', 'brand', 
              'inUse', 'externalCal']

for col in label_cols:
    df[col] = df[col].astype('category')  # Convert to category dtype
    df[col] = df[col].cat.codes  # Convert categorical to numerical codes

# Convert calibrationInterval to numeric (if applicable), handling errors
df['calibrationInterval'] = pd.to_numeric(df['calibrationInterval'], errors='coerce')

# Drop non-numeric columns that won't be useful for training
drop_columns = ['tag', 'modelPartNo', 'serialIdNo', 'range', 'externalToleranceLimit', 
                'internalTolerenceLimit', 'calibrationReportNumber', 'calibrator', 'pic', 
                'actionForRenewalReminder', 'lastCalibration', 'calibrationDue', 'status']
df.drop(columns=drop_columns, inplace=True)

# Feature set (X) and target variable (y)
X = df.drop(columns=['remainingMths'])
y = df['remainingMths']

# Convert target variable to numeric, coercing errors to NaN
y = pd.to_numeric(y, errors='coerce')

# Handle any NaN values in the target variable
X = X[~y.isna()]
y = y.dropna()

# Save the column order
joblib.dump(X.columns.tolist(), 'column_order.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix format (XGBoost’s internal data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',  # Regression problem
    'eval_metric': 'rmse',  # Use RMSE to evaluate performance
    'max_depth': 5,  # Limit the tree depth to prevent overfitting
    'learning_rate': 0.01,  # Lower learning rate
    'n_estimators': 1000,  # Increase number of trees
    'lambda': 1.0,  # L2 regularization
    'alpha': 0.5,  # L1 regularization
    'min_child_weight': 5,  # Prevent overfitting on small datasets
    'subsample': 0.8,  # Use 80% of data for each tree
    'colsample_bytree': 0.8  # Use 80% of features for each tree
}

# Train the model
model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=50, evals=[(dtrain, 'train'), (dtest, 'eval')])

# Predict on the test set
predictions = model.predict(dtest)

# Evaluate model performance
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save the trained XGBoost model to a file
model.save_model("xgboost_model.json")
print("Model saved successfully!")

# Check evaluation results during training
evals_result = {}
model = xgb.train(
    params, dtrain, num_boost_round=1000, early_stopping_rounds=50, 
    evals=[(dtrain, 'train'), (dtest, 'eval')], evals_result=evals_result
)

print(evals_result)
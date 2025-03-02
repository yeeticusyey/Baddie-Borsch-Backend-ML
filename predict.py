import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os
import numpy as np

# Load the trained model
loaded_model = xgb.Booster()
loaded_model.load_model("xgboost_model.json")

# Load the column order
column_order = joblib.load('column_order.pkl')

# Parse the JSON data from the file (adjust the path to your actual JSON file)
cwd = os.getcwd()
json_file_path = os.path.join(cwd, "data", "Bosch_Dataset.json")
new_json_file_path = os.path.join(cwd, "data", "Bosch_Dataset_Predictions.json")
new_excel_file_path = os.path.join(cwd, "data", "Bosch_Dataset_Predictions.xlsx")

# Ensure the JSON file is correctly formatted
with open(json_file_path, 'r') as file:
    try:
        data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        exit(1)

# Convert JSON data to pandas DataFrame
test_df = pd.DataFrame(data)

# Store the original 'lastCalibration' and 'calibrationDue' dates
original_last_calibration = pd.to_datetime(test_df['lastCalibration'], errors='coerce')
original_calibration_due = pd.to_datetime(test_df['calibrationDue'], errors='coerce')
serial_id_no = test_df['serialIdNo'].copy()  # Store the serialIdNo

# Drop unnecessary columns (those not used in the model)
test_df = test_df.drop(columns=['tag', 'modelPartNo', 'serialIdNo', 'range', 'externalToleranceLimit', 
                                'internalTolerenceLimit', 'calibrationReportNumber', 'calibrator', 'pic', 
                                'actionForRenewalReminder'])

# Preprocess categorical columns just like in training
label_cols = ['div', 'description', 'standard', 'category', 'brand', 'inUse', 'externalCal']

# Initialize LabelEncoder
label_encoders = {}

# Encode categorical columns
for col in label_cols:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col].astype(str))
    label_encoders[col] = le  # Save encoder for future use (if needed)

# Feature engineering: calculate days until calibration due
test_df['lastCalibration'] = pd.to_datetime(test_df['lastCalibration'], errors='coerce')
test_df['calibrationDue'] = pd.to_datetime(test_df['calibrationDue'], errors='coerce')

# Calculate 'Days Since Last Calibration' and 'Days Until Calibration Due'
test_df['lastCalibration_days'] = (test_df['lastCalibration'] - pd.to_datetime('1970-01-01')).dt.days
test_df['calibrationDue_days'] = (test_df['calibrationDue'] - pd.to_datetime('1970-01-01')).dt.days

# Calculate 'Days Until Calibration Due'
test_df['daysUntilCalibrationDue'] = (test_df['calibrationDue'] - test_df['lastCalibration']).dt.days

# Ensure 'calibrationInterval' is in the test data and it is placed last
test_df['calibrationInterval'] = pd.to_numeric(test_df['calibrationInterval'], errors='coerce')

# Handle any NaN values in the test data
test_df = test_df.fillna(0)

# Reorder columns to match the exact order from the training set
test_df = test_df[column_order]

# Convert to DMatrix format (XGBoost format)
dnew = xgb.DMatrix(test_df)

# Make predictions
predictions = loaded_model.predict(dnew)

# Output the predictions
print("Predictions:", predictions)

# Filter out NaN predictions
valid_indices = ~np.isnan(predictions)
predictions = predictions[valid_indices]
original_last_calibration = original_last_calibration[valid_indices]
original_calibration_due = original_calibration_due[valid_indices]
serial_id_no = serial_id_no[valid_indices]

# Now we use the predicted months to calculate the predicted calibration date for each row
predicted_dates = []
for i, predicted_months in enumerate(predictions):
    last_calibration_date = pd.to_datetime(original_last_calibration.iloc[i], errors='coerce')
    predicted_months_int = int(predicted_months)  # Convert the predicted months to an integer
    predicted_date = last_calibration_date + pd.DateOffset(months=predicted_months_int)
    predicted_dates.append(predicted_date)

# Update the original data with new fields
for i, predicted_date in enumerate(predicted_dates):
    actual_due_date = pd.to_datetime(original_calibration_due.iloc[i], errors='coerce')
    difference = (predicted_date - actual_due_date).days  # Calculate the difference in days
    serial_id = serial_id_no.iloc[i]  # Get the serialIdNo for the current row
    prediction_value = float(predictions[i])  # Convert to float for JSON serialization
    
    # Update the original data with new fields
    data[i]['predictedIdealCalibrationDate'] = str(predicted_date)
    data[i]['differenceFromPredictions'] = difference
    data[i]['predictionValue'] = prediction_value

# Save the updated JSON data to a new file
with open(new_json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

# Convert the updated data to a DataFrame
output_df = pd.DataFrame(data)

# Save the output DataFrame to an Excel file
output_df.to_excel(new_excel_file_path, index=False)

print(f"Updated JSON data has been saved to {new_json_file_path}")
print(f"Updated Excel data has been saved to {new_excel_file_path}")
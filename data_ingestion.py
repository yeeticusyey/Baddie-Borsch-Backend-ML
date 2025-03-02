import pandas as pd 
import numpy as np
import os
import json

# Load the data
cwd = os.getcwd()
dataset_dir = os.path.join(cwd, "data", "Cleaned_Bosch_Dataset.xlsx")

dfs = pd.read_excel(dataset_dir, sheet_name=None)
excel_data_fragment = pd.read_excel(dataset_dir, sheet_name='Cleaned_Bosch_Dataset')

# Remove rows where all values are NaN
# excel_data_fragment.dropna(how='all', inplace=True)

# Convert datetime objects to strings
for column in excel_data_fragment.select_dtypes(include=['datetime64']).columns:
    excel_data_fragment[column] = excel_data_fragment[column].astype(str)

# Calculate the status based on the next calibration due date
current_date = pd.Timestamp.now()
excel_data_fragment['calibrationDue'] = pd.to_datetime(excel_data_fragment['calibrationDue'], errors='coerce')
excel_data_fragment['status'] = excel_data_fragment['calibrationDue'].apply(
    lambda due_date: 'Overdue' if (due_date - current_date).days < 0 else 
                     'Drifting' if (due_date - current_date).days <= 90 else 
                     'Optimal'
)

# Convert each row to a dictionary with column headers as keys
json_data = excel_data_fragment.astype(str).to_dict(orient='records')

# Save JSON data to a file
json_file_path = os.path.join(cwd, "data", "Bosch_Dataset.json")
with open(json_file_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print('Excel Sheet to JSON:\n', json_data)
print(f'JSON data has been saved to {json_file_path}')
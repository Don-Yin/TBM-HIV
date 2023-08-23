import pandas as pd
from pathlib import Path
from src.select_variables import selected_variables

from rich import print

# Load data
data = pd.read_csv(Path("data", "2_clinical_only_selected", "clinical.csv"))

# Obtain unique project_opfu values
project_opfu_values = data["project"].unique()

# Create a dictionary to store missing data information
missing_data_info = {}

variable_names = [i.data_name_in_merged.lower() for i in selected_variables]

for var in variable_names:
    missing_data_info[var] = {}
    for project in project_opfu_values:
        project_data = data[data["project"] == project]
        missing_data_info[var][project] = project_data[var].isnull().mean()

missing_data_df = pd.DataFrame.from_dict(missing_data_info)

missing_indicator = data.isnull().astype(int)

correlation_matrix = missing_indicator.corr()

systematic_missing_pairs = correlation_matrix[correlation_matrix > 0.9].stack().reset_index()
systematic_missing_pairs = systematic_missing_pairs.query("level_0 < level_1")


data_filled = data.fillna(-1)

for col in data_filled.columns:
    if data_filled[col].dtype == "object":
        data_filled[col] = data_filled[col].astype("category").cat.codes

correlation_matrix_values = data_filled.corr()

systematic_missing_values_pairs = correlation_matrix_values[correlation_matrix_values > 0.9].stack().reset_index()
systematic_missing_values_pairs = systematic_missing_values_pairs.query("level_0 < level_1")

print("\n---------- Missing data information ----------\n\n")
print(systematic_missing_pairs)
print(systematic_missing_values_pairs)
print("\n\n---------- End Missing data information ----------\n")

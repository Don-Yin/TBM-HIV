import pandas
import os
from pathlib import Path
from src.select_variables import selected_variables
from rich import print

merged_data_path = Path("data", "1_clinical_merged", "clinical.csv")
merged_dataframe = pandas.read_csv(merged_data_path)

# only keep the columns that are in the selected variables + subject ID + project_opfu
merged_dataframe = merged_dataframe[
    ["subject_id", "project_opfu"] + [i.data_name_in_merged.lower() for i in selected_variables]
]

# rename the project_opfu column to project
merged_dataframe.rename(columns={"project_opfu": "project"}, inplace=True)

# check amount of missing values for project
print("------ Amount of missing values for in the merged dataframe ------")
print(merged_dataframe["project"].isna().sum())

# read the clinical data
clinical_data_path = Path("data", "clinical")
clinical_files = os.listdir(clinical_data_path)

# Loop over each row in the merged_dataframe
for idx, row in merged_dataframe.iterrows():
    # If the project is NaN
    if pandas.isna(row["project"]):
        # Get the subject_id
        subject_id = row["subject_id"]

        # Loop over each file in the clinical_data_path
        for file in clinical_files:
            # Read the file
            df = pandas.read_csv(clinical_data_path / file)

            # Check if the subject_id is in the file
            if subject_id in df["subject_id"].values:
                # Get the project of the subject_id
                project = df.loc[df["subject_id"] == subject_id, "project"].values[0]

                # If project is not NaN
                if not pandas.isna(project):
                    # Update the project in the merged_dataframe
                    merged_dataframe.loc[idx, "project"] = project
                    # Break the loop as we found the project
                    break

# Verify if there are still missing values for project
print(merged_dataframe["project"].isna().sum())
print("------ Amount of missing values for in the merged dataframe ------\n\n")

# save the filtered dataframe
merged_dataframe.to_csv(Path("data", "2_clinical_only_selected", "clinical.csv"), index=False)

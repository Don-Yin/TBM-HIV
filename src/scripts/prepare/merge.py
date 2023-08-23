from pathlib import Path
import pandas
import os
from functools import reduce
from src.clinical.modules.data_classes import ClinicalData


# clinical data --------------------------------------------
data_path = Path("data", "clinical")
data_files = os.listdir(data_path)
data_files = [i for i in data_files if i.endswith(".csv")]
data_objects = [ClinicalData(data_path / i) for i in data_files]

for i in data_objects:
    i.check_num_unique_subjects()

# merge use column subject_id
merged_dataframe = reduce(
    lambda df1, df2: pandas.merge(df1, df2, on="subject_id", how="outer"),
    [i.dataframe for i in data_objects],
)
merged_dataframe.to_csv(Path("data", "1_clinical_merged", "clinical.csv"), index=False)

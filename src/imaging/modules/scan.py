import os
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted




class Scan:
    def __init__(self, name):
        path_label = Path("data", "MRI_sessions.csv")
        frame_label = pd.read_csv(path_label)

        self.file_name = name
        self.label_frame = frame_label

        self.mr_id = self.image_name_to_mr_id()
        self.subject_id = self.image_name_to_subject_id()
        self.date = self.image_name_to_date()
        self.lesion_type = self.check_lesion_type()

    def image_name_to_mr_id(self):
        return "_".join(self.file_name.split("_")[:2])

    def image_name_to_subject_id(self):
        mr_id = self.image_name_to_mr_id()
        return self.id_conversion(mr_id, "subject")

    def image_name_to_date(self):
        mr_id = self.image_name_to_mr_id()
        row = self.label_frame[self.label_frame["MR ID"] == mr_id]
        return pd.to_datetime(row["Date"].values[0], format="%d/%m/%Y")

    def id_conversion(self, num, target_system):
        if target_system == "subject":
            return self.label_frame.loc[self.label_frame["MR ID"] == num, "subject_id"].values[0]
        elif target_system == "mr":
            return self.label_frame.loc[self.label_frame["subject_id"] == num, "MR ID"].values[0]

    def check_lesion_type(self):
        mr_id = self.image_name_to_mr_id()
        row = self.label_frame[self.label_frame["MR ID"] == mr_id]
        value = str(row["Type"].values[0])
        if value != "nan":
            return int(value[0])
        return np.nan


if __name__ == "__main__":
    pass

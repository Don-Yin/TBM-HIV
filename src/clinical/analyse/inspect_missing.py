"""This script is used to inspect the locally csv data files"""
import pandas
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.random_forest import selected_features
import numpy as np


def inspect_csv_file(file_path, output_path=None, selected_variables=None):
    if selected_variables:
        interested_columns = [i.data_name_in_merged for i in selected_variables]
        interested_columns = [i.lower() for i in interested_columns]
    else:
        interested_columns = None

    if output_path is None:
        save_path = Path("results", "images", "missing_values")
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = output_path

    frame = pandas.read_csv(file_path)
    frame.columns = frame.columns.str.lower()

    if interested_columns:
        frame = frame[interested_columns]

    # -------- figure bit --------

    plt.figure(figsize=(10, 6))
    sns.heatmap(frame.isna().transpose(), cmap="YlGnBu", cbar_kws={"label": "Missing Data"})

    plt.tight_layout()
    plt.savefig(save_path / (file_path.stem + ".png"))
    plt.close()

    # -------- text bit --------

    missing_value_percentages = []
    for column in frame.columns:
        # print(frame[column].value_counts())
        na_list = frame[column].isna().to_list()
        na_len = len([i for i in na_list if i])
        total_len = len(frame[column])
        missing_value_percentage = (na_len / total_len) * 100
        missing_value_percentages.append(missing_value_percentage)

    save_text = f"""Among the total curated variables:
    max_missing_percentage: {max(missing_value_percentages)}%
    mean_missing_percentage: {np.mean(missing_value_percentages)}%
    median_missing_percentage: {np.median(missing_value_percentages)}
    """

    with open(save_path / "missing_values.MD", "w") as writer:
        writer.write(save_text)


if __name__ == "__main__":
    path_to_data = Path("data", "2_clinical_only_selected")
    files = os.listdir(path_to_data)
    files = [i for i in files if i.endswith(".csv")]

    for file in files:
        inspect_csv_file(path_to_data / file)

"""
this script prepares the clinical data for the deep learning model
"""

from src.random_forest import selected_features
from sklearn.preprocessing import MinMaxScaler
import torch
from pathlib import Path
import random
import pandas
import json
import os

# ---------------- train test split subjects ----------------
with open(Path("cache", "split.json"), "r") as loader:
    split_dict = json.load(loader)
    train_subject_ids = split_dict["train"]
    test_subject_ids = split_dict["test"] + split_dict["valid"]

# ---------------- getting the imputated data ----------------
imputation_data_folder = Path("data", "3_clinical_imputations")
imputation_data_files = [i for i in os.listdir(imputation_data_folder) if i.endswith(".pkl")]


# ---------------- reading a sample file ----------------
def parse_sample(path_to_sample: Path):
    keep_indices = ["subject_id", "project", "tbmgrade_followupsigns"] + selected_features
    dataframe = pandas.read_pickle(path_to_sample)
    dataframe = dataframe[keep_indices]
    return dataframe


def filter_subject(dataframe, which: str):
    subject_ids = train_subject_ids if which == "train" else test_subject_ids
    return dataframe[dataframe["subject_id"].isin(subject_ids)]


# ---------------- recode ----------------
recode_dicts = {
    "gender_base": {"Female": 0, "Male": 1},
    "paraplegia_clinical": {"Yes": 1, "No": 0},
    "urinaryretention_clinical": {"Yes": 1, "No": 0},
    "requirehelp_opfu": {"Yes": 1, "No": 0},
    "disproblem_opfu": {"Yes": 1, "No": 0},
    "weightloss_clinical": {"Yes": 1, "No": 0},
}

samples = [parse_sample(imputation_data_folder / i) for i in imputation_data_files]  # all imputed data


def recode(dataframe):
    """
    recode the categorical variables
    """
    for i in recode_dicts:
        # if i not in dataframe.columns:
        #     continue
        dataframe[i] = dataframe[i].map(recode_dicts[i])
        dataframe[i] = dataframe[i].astype("int")
    return dataframe


samples = [recode(i) for i in samples]

# check all columns in all samples are numbers
for i in samples:
    non_numeric_columns = i.select_dtypes(exclude="number").columns
    non_numeric_columns = [i for i in non_numeric_columns if i not in ["subject_id", "project"]]
    non_numeric_columns_values = {j: list(i[j].unique()) for j in non_numeric_columns}
    error_message = (
        f"There are non-numeric columns in the samples {non_numeric_columns}; with values {non_numeric_columns_values}"
    )
    assert len(non_numeric_columns) == 0, error_message


# ---------------- normalize ----------------
def normalize(dataframe):
    """
    Normalizing and splitting has to be done at the same time since the norm stats has to be calculated from the training data and applied on the test data
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))

    X_train = filter_subject(dataframe, "train")
    X_test = filter_subject(dataframe, "test")

    # ---- keep only selected variables ----
    X_train = X_train.drop(["subject_id", "project", "tbmgrade_followupsigns"], axis=1)
    X_test = X_test.drop(["subject_id", "project", "tbmgrade_followupsigns"], axis=1)

    numeric_columns = X_train.select_dtypes(include="number").columns
    scaler.fit(X_train[numeric_columns])

    # ---- apply the normalizations ----
    X_train[numeric_columns] = scaler.transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

    # ---- reorder the columns to be the same as the selected features ----
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # ---- put the subject_id and project back at the front ----
    X_train["subject_id"] = filter_subject(dataframe, "train")["subject_id"]
    X_train["project"] = filter_subject(dataframe, "train")["project"]
    X_test["subject_id"] = filter_subject(dataframe, "test")["subject_id"]
    X_test["project"] = filter_subject(dataframe, "test")["project"]

    # move to the front
    X_train = X_train[["subject_id", "project"] + list(X_train.columns[:-2])]
    X_test = X_test[["subject_id", "project"] + list(X_test.columns[:-2])]
    assert list(X_train.columns) == list(X_test.columns), "The columns of the train and test data are not the same"

    # combine the train and test data to one
    combined = pandas.concat([X_train, X_test])

    return X_train, X_test, combined


# ---------------- make variables ----------------

samples = [normalize(i) for i in samples]


# ---------------- export ----------------
def get_clinical_data(subject_id):
    """
    get the clinical data of a subject by id
    """
    _, _, combined = random.choice(samples)
    row = combined[combined["subject_id"] == subject_id]
    row = row.drop(["subject_id", "project"], axis=1)
    row = torch.tensor(row.values).float()
    # remove the batch dimension
    row = row.squeeze()
    return row


if __name__ == "__main__":
    get_clinical_data("BMEIS_S07545")

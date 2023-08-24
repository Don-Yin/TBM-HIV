"""
this script is for feature selection (identify the clinical bio markers)
Input: imputed datasets (csv or pickle)
Output: selected variable by feature importance (export); feature importance plot (df and plot)
double check this needs to be done on each of the imputed dataframes
report R2 as a measure of the accountablity of the biomarkers?
"""

import json
import os
import pandas
import random
from rich import print
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from src.select_variables import variables

with open(Path("cache", "split.json"), "r") as loader:
    split_dict = json.load(loader)
    train_subject_ids = split_dict["train"]
    test_subject_ids = split_dict["test"] + split_dict["valid"]

imputation_data_folder = Path("data", "3_clinical_imputations")
imputation_data_files = [i for i in os.listdir(imputation_data_folder) if i.endswith(".pkl")]

sample = random.choice(imputation_data_files)
sample = Path(imputation_data_folder, sample)
sample = pandas.read_pickle(sample)

# identify numerical and categorical columns
num_cols = sample.select_dtypes(include=["int64", "float64"]).columns
num_cols = [i for i in num_cols if i not in ["subject_id", "project", "tbmgrade_followupsigns"]]
cat_cols = sample.select_dtypes(include=["category"]).columns

# create transformer for numerical and categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols),
    ]
)


def get_importance_r2():
    """
    The main function that calculates the feature importance and R2 score across the impuated datasets
    """
    all_importances = []
    r2_scores = []

    # compute using all imputed datasets
    for file in imputation_data_files:
        sample = Path(imputation_data_folder, file)
        sample = pandas.read_pickle(sample)

        X = sample.drop(["tbmgrade_followupsigns", "subject_id", "project"], axis=1)
        y = sample["tbmgrade_followupsigns"]

        # Split the data based on subject IDs
        X_train = X[sample["subject_id"].isin(train_subject_ids)]
        y_train = y[sample["subject_id"].isin(train_subject_ids)]
        X_test = X[sample["subject_id"].isin(test_subject_ids)]
        y_test = y[sample["subject_id"].isin(test_subject_ids)]

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train_transformed, y_train)

        y_pred = clf.predict(X_test_transformed)
        r2_scores.append(r2_score(y_test, y_pred))

        importances = clf.feature_importances_
        all_importances.append(importances)

    across_sets_importances = np.mean(all_importances, axis=0)
    across_sets_r2_scores = np.mean(r2_scores)
    # print(f"Average R2 across imputed dataset with all features {across_sets_r2_scores}")
    return across_sets_importances, across_sets_r2_scores


def get_importance_df():
    across_sets_importances, _ = get_importance_r2()
    one_hot_columns = preprocessor.named_transformers_["cat"].get_feature_names_out(input_features=cat_cols)
    importances_df = pandas.DataFrame({"Feature": np.append(num_cols, one_hot_columns), "Importance": across_sets_importances})
    importances_df.sort_values(by="Importance", ascending=False, inplace=True)
    return importances_df


def draw_elbow_curve(feature_counts, r2_scores):
    plt.plot(feature_counts, r2_scores, "b*-")
    plt.xlabel("Number of Features")
    plt.ylabel("R-squared Score")
    plt.title("Elbow Curve - R-squared Score vs Number of Features")
    plt.savefig(Path("results", "images", "random_forest_elbow_curve.png"))


def draw_elbow_curve_main():
    sample = Path(imputation_data_folder, random.choice(imputation_data_files))
    sample = pandas.read_pickle(sample)

    X = sample.drop(["tbmgrade_followupsigns", "subject_id", "project"], axis=1)
    y = sample["tbmgrade_followupsigns"]

    X_train = X[sample["subject_id"].isin(train_subject_ids)]
    y_train = y[sample["subject_id"].isin(train_subject_ids)]
    X_test = X[sample["subject_id"].isin(test_subject_ids)]
    y_test = y[sample["subject_id"].isin(test_subject_ids)]

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    num_features = X_train_transformed.shape[1]
    r2_scores = []
    feature_counts = []

    for i in range(num_features):
        r2_scores_across_imputation = []

        for file in imputation_data_files:
            sample = Path(imputation_data_folder, file)
            sample = pandas.read_pickle(sample)

            X = sample.drop(["tbmgrade_followupsigns", "subject_id", "project"], axis=1)
            y = sample["tbmgrade_followupsigns"]

            X_train = X[sample["subject_id"].isin(train_subject_ids)]
            y_train = y[sample["subject_id"].isin(train_subject_ids)]
            X_test = X[sample["subject_id"].isin(test_subject_ids)]
            y_test = y[sample["subject_id"].isin(test_subject_ids)]

            clf = RandomForestClassifier(n_estimators=100, random_state=i)
            clf.fit(X_train_transformed, y_train)

            y_pred = clf.predict(X_test_transformed)
            r2_scores_across_imputation.append(r2_score(y_test, y_pred))

        # Remove the least important feature for the next iteration
        least_important_feature_index = np.argmin(clf.feature_importances_)
        X_train_transformed = np.delete(X_train_transformed, least_important_feature_index, axis=1)
        X_test_transformed = np.delete(X_test_transformed, least_important_feature_index, axis=1)

        r2_scores.append(np.mean(r2_scores_across_imputation))
        feature_counts.append(num_features - i)

    corresponding_feature_r2 = {f + 1: r for f, r in zip(feature_counts, r2_scores)}

    # get the key value when the value is the highest
    max_r2 = max(corresponding_feature_r2.values())
    max_feature_count = max(corresponding_feature_r2, key=corresponding_feature_r2.get)
    print(f"Max R2 score: {max_r2} with {max_feature_count} features")
    draw_elbow_curve(feature_counts, r2_scores)


def final_select_features():
    """
    Currently it selects by cutting from the median, but can be amended.
    """
    importances_df = get_importance_df()

    # median_importance = importances_df["Importance"].median()
    # selected_features = importances_df[importances_df["Importance"] > median_importance]["Feature"].values

    # select the top 48 most important features
    selected_features = importances_df["Feature"].values[:56]
    selected_features_importance = importances_df["Importance"].values[:56]
    selected_features_importance = [round(i, 4) for i in selected_features_importance]

    # Calculate the mean R2 scores with the selected features across imputed datasets
    r2_scores = []

    for file in imputation_data_files:
        sample = Path(imputation_data_folder, file)
        sample = pandas.read_pickle(sample)

        X = sample.drop(["tbmgrade_followupsigns", "subject_id", "project"], axis=1)
        y = sample["tbmgrade_followupsigns"]

        X_train = X[sample["subject_id"].isin(train_subject_ids)]
        y_train = y[sample["subject_id"].isin(train_subject_ids)]
        X_test = X[sample["subject_id"].isin(test_subject_ids)]
        y_test = y[sample["subject_id"].isin(test_subject_ids)]

        # Transform X_train and X_test
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Convert to DataFrame to support indexing by column name
        ohe_features = preprocessor.named_transformers_["cat"].get_feature_names_out(input_features=cat_cols)
        all_features = np.append(num_cols, ohe_features)
        X_train_transformed = pandas.DataFrame(X_train_transformed, columns=all_features)
        X_test_transformed = pandas.DataFrame(X_test_transformed, columns=all_features)

        # Select features
        X_train_transformed = X_train_transformed[selected_features]
        X_test_transformed = X_test_transformed[selected_features]

        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train_transformed, y_train)

        y_pred = clf.predict(X_test_transformed)
        r2_scores.append(r2_score(y_test, y_pred))

    mean_r2_score = np.mean(r2_scores)
    save_content = (
        f"R2 scores using the selected features across different imputated dataframes: {r2_scores}; mean: {mean_r2_score}"
    )

    save_path = Path("results", "images", "random_forest_elbow_curve.MD")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as writer:
        writer.write(save_content)

    # print(f"R2 in the retrained model: {mean_r2_score}")
    return selected_features, selected_features_importance


def parse_onehot_encoded_features(selected_featuers):
    """
    parsing the selected featues back to before one-hot encoding
    """
    sample = Path(imputation_data_folder, random.choice(imputation_data_files))
    sample = pandas.read_pickle(sample)

    for i in range(len(selected_featuers)):
        if not selected_featuers[i] in sample.columns:
            selected_featuers[i] = "_".join(selected_featuers[i].split("_")[:-1])

    assert all([i in sample.columns for i in selected_featuers]), "Not all selected features are in the original dataset"

    return list(selected_featuers)


# print("------------------------------------------------------------------------")
selected_features, selected_features_importance = final_select_features()
selected_features = parse_onehot_encoded_features(selected_features)
variables.plot_table(selected_features)
# print("------------------------------------------------------------------------")

# --------- drop gcs_total_followupsigns / gcs_verbal_followupsigns
# selected_features = [i for i in selected_features if not "gcs" in i]


if __name__ == "__main__":
    # print("------------------------------------------------------------------------")
    corresponding_feature_importance = {f: i for f, i in zip(selected_features, selected_features_importance)}
    print(corresponding_feature_importance)
    # draw_elbow_curve_main()
    # selected_features = final_select_features()
    print("------------------------------------------------------------------------")

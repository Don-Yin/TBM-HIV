from pathlib import Path
import os
from src.imaging.modules.scan import Scan
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np


def get_use_subjects():
    # if this fails refer to the image registry at cache; not published due to patient privacy; cache contains patient id
    path_images = Path("/Users/donyin/Desktop/images_only_first")
    folders = os.listdir(path_images)
    folders = [i for i in folders if not i.startswith(".")]

    use_subjects = []
    for folder in folders:
        target_folder = path_images / folder
        images = os.listdir(target_folder)
        scans = [Scan(i) for i in images]
        subject_ids = [i.subject_id for i in scans]
        use_subjects += subject_ids

    return use_subjects


# -------- conversion --------
def subject_id_to_tbm_grade(subject_id):
    clinical_frame = pandas.read_csv(Path("data", "1_clinical_merged", "clinical.csv"))
    row_clinical = clinical_frame.loc[clinical_frame["subject_id"] == subject_id]
    tbm_grade = row_clinical["tbmgrade_followupsigns"].values[0]
    return int(tbm_grade)  # -1


def subject_id_to_lesion_type(subject_id):
    # Read the csv file
    label_frame = pandas.read_csv(Path("data", "MRI_sessions.csv"))
    filtered_frame = label_frame[label_frame["subject_id"] == subject_id].copy()  # make a copy here
    filtered_frame["Date"] = pandas.to_datetime(filtered_frame["Date"], format="%d/%m/%Y")
    mr_id = filtered_frame.sort_values(by="Date").iloc[0]["MR ID"]

    # Extract the label value
    row = label_frame.loc[label_frame["MR ID"] == mr_id]
    label_str = row["Type"].values[0]  # e.g., '2 - Granulomas, more than 5' / nan

    assert not pandas.isnull(label_str), f"label_str is nan for MR ID {mr_id}"
    label_value = str(label_str[0])  # e.g., 2
    return label_value


if __name__ == "__main__":
    save_to = Path("results", "images", "demographics")
    save_content = ""
    data = [
        {"subject_id": i, "tbm_grade": subject_id_to_tbm_grade(i), "lesion_type": subject_id_to_lesion_type(i)}
        for i in get_use_subjects()
    ]
    data = pandas.DataFrame(data)
    # the lesion type is categotical, the tbm grade is ordinal

    contingency_table = pandas.crosstab(data["tbm_grade"], data["lesion_type"])

    # the chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)
    save_content += f"Chi-squared: {chi2:.4f}\n"
    save_content += f"P-value: {p:.4f}\n"

    # -------- cramer's v --------
    n = contingency_table.sum().sum()
    k = contingency_table.shape[1]
    r = contingency_table.shape[0]
    cramers_v = np.sqrt(chi2 / (n * min(k - 1, r - 1)))
    save_content += f"Cramér's V: {cramers_v:.4f}\n"

    # -------- plot --------
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, cmap="Blues", fmt="g")
    plt.title("Contingency table between TBM Grade and Lesion Type")
    plt.ylabel("TBM Grade")
    plt.xlabel("Lesion Type")
    plt.savefig(save_to / "tbm_lesion_contingency_table.png")

    save_content += """\n
    since, lesion_type is nominal (not ordinal), then computing correlation becomes more complicated, need to resort to some form of contingency table analysis like the chi-squared test or Cramér's V.

    The p-value:
        The p-value (0.0008 in this case) represents the probability of observing the given data (or something more extreme) if the null hypothesis is true. The null hypothesis for the chi-squared test is that there's no association between the two categorical variables (tbm_grade and lesion_type).

        Given that the p-value is 0.0008, which is less than the common significance level of 0.05, we reject the null hypothesis. This means we have sufficient evidence to conclude that there's a significant association between tbm_grade and lesion_type.

    The chi-squared statistic (χ²):
        The chi-squared statistic (30.2245 in this case) represents the difference between the observed frequencies in your data and the frequencies you would expect under the assumption of independence (i.e., if tbm_grade and lesion_type were not related at all).

        A larger χ² statistic indicates a larger discrepancy between the observed and expected frequencies. If the two categorical variables are independent (i.e., not associated), you would expect the χ² value to be small. If they are associated, the χ² value would be large.

    Cramér's V is a statistic derived from the chi-squared statistic that provides a measure of association between two nominal variables. It ranges from 0 (no association) to 1 (perfect association).
    """

    with open(save_to / "tbm_lesion_contingency_table.MD", "w") as loader:
        loader.write(save_content)

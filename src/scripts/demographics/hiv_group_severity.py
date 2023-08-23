"""
this is just for demographics

this script checks for the group HIV severity
this is done by check the project variable OUCRU_TBM26 / 27 on the distribution of TBMGrade
26 is the HIV+ group
27 is the HIV- group

"""

import pandas
from pathlib import Path
from src.select_variables import variables
import matplotlib.pyplot as plt
import seaborn as sns
from src.imaging.modules.scan import Scan
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def combined_analysis(data):
    tbm_data_name = "tbmgrade_followupsigns"

    save_path = Path("results", "images", "report", "hiv_group_severity.MD")
    save_content = ""

    # 1. Visualization
    sns.boxplot(x="project", y=tbm_data_name, data=data)
    plt.title("Distribution of TBM Grade by Project")

    # Separate data into groups based on project
    group_26 = data[data["project"] == "OUCRU_TBM26"][tbm_data_name].dropna()
    group_27 = data[data["project"] == "OUCRU_TBM27"][tbm_data_name].dropna()

    # 2. Check for normality using Shapiro-Wilk test
    _, p_26 = stats.shapiro(group_26)
    _, p_27 = stats.shapiro(group_27)

    alpha = 0.05
    if p_26 < alpha and p_27 < alpha:
        # Non-parametric approach
        print("As per Shapiro-Wilk test, data is not normally distributed, using Mann-Whitney U test")
        save_content += "Data is not normally distributed, using Mann-Whitney U test\n"
        u_statistic, p_value = stats.mannwhitneyu(group_26, group_27)

        # Calculate effect size for Mann-Whitney
        n1 = len(group_26)
        n2 = len(group_27)
        r = u_statistic / (n1 * n2)
    else:
        # Parametric approach
        print("As per Shapiro-Wilk test, data is normally distributed, using independent t-test")
        save_content += "Data is normally distributed, using independent t-test\n"
        t_statistic, p_value = stats.ttest_ind(group_26, group_27)

        # Calculate effect size for t-test (Cohen's d)
        pooled_std = (((n1 - 1) * group_26.std() ** 2 + (n2 - 1) * group_27.std() ** 2) / (n1 + n2 - 2)) ** 0.5
        d = abs(group_26.mean() - group_27.mean()) / pooled_std
        r = d / ((d**2 + 4) ** 0.5)  # Convert Cohen's d to r

    print(f"P-value = {p_value:.5f}")
    print(f"Effect size (r) = {r:.3f}")
    save_content += f"P-value = {p_value:.5f}\n"
    save_content += f"Effect size (r) = {r:.3f}\n"
    save_content += f"U-statistic = {u_statistic:.3f}\n"
    n = len(group_26) + len(group_27)

    # Interpretation
    if p_value < alpha:
        print(f"There's a statistically significant difference in TBM Grades between the two projects (p = {p_value:.5f}).")
        save_content += (
            f"There's a statistically significant difference in TBM Grades between the two projects (p = {p_value:.5f}).\n"
        )
    else:
        print(f"There's no statistically significant difference in TBM Grades between the two projects (p = {p_value:.5f}).")
        save_content += (
            f"There's no statistically significant difference in TBM Grades between the two projects (p = {p_value:.5f}).\n"
        )

    # Note about effect size
    print(
        "\nNote: Regardless of the p-value, it's always important to consider the effect size when interpreting the results, as it provides insight into the practical significance of the observed difference."
    )
    save_content += "\nNote: Regardless of the p-value, it's always important to consider the effect size when interpreting the results, as it provides insight into the practical significance of the observed difference.\n"
    save_content += f"\nn = {n}\n"

    # 3. Save to file
    with open(save_path, "w") as f:
        f.write(save_content)


# --------
# if this fails refer to the image registry at cache
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

# --------


save_to = Path("results", "images", "demographics")
save_to.mkdir(parents=True, exist_ok=True)

tbm_grade = variables.retrieve("tbmgrade")
tbm_data_name = tbm_grade.data_name_in_merged

data = pandas.read_csv(Path("data", "2_clinical_only_selected", "clinical.csv"))
data = data[["subject_id", "project", tbm_data_name]]

# Filter out subjects that are not in the images folder
data = data[data["subject_id"].isin(use_subjects)]

combined_analysis(data)

data_OUCRU_TBM26 = data[data["project"] == "OUCRU_TBM26"]
data_OUCRU_TBM27 = data[data["project"] == "OUCRU_TBM27"]

TBM_26_value_counts = data_OUCRU_TBM26[tbm_data_name].value_counts()
TBM_27_value_counts = data_OUCRU_TBM27[tbm_data_name].value_counts()

plot_data = pandas.DataFrame(
    {
        "tbmgrade": TBM_26_value_counts.index.tolist() + TBM_27_value_counts.index.tolist(),
        "count": TBM_26_value_counts.tolist() + TBM_27_value_counts.tolist(),
        "project": ["OUCRU_TBM26"] * len(TBM_26_value_counts) + ["OUCRU_TBM27"] * len(TBM_27_value_counts),
    }
)

# Plot the distribution using seaborn
plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=plot_data, x="tbmgrade", y="count", hue="project")
plt.title("Distribution of TBMGrade for HIV+ and HIV- groups", fontsize=16)
plt.xlabel("TBM Grade", fontsize=14)
plt.ylabel("Count", fontsize=14)

ymax = max(TBM_26_value_counts.max(), TBM_27_value_counts.max()) * 1.10
plt.ylim(0, ymax)

handles, labels = barplot.get_legend_handles_labels()
labels[0] = "HIV+"
labels[1] = "HIV-"
plt.legend(handles, labels, title="Group", fontsize=12)


# Adding labels on top of the bars
for p in barplot.patches:
    barplot.annotate(
        format(p.get_height(), ".0f"),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 9),
        textcoords="offset points",
        fontsize=14,
    )

plt.tight_layout()
plt.savefig(save_to / "TBMGrade_distribution.png", dpi=420)

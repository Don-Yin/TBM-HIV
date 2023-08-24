"""
analyse the results produced by inferencing the test set
the information here is unique and cannot be obtained from weights and biaes or tensorboard
"""
from natsort import natsorted
from rich import print
from pathlib import Path
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from src.analysis.confusion_matrices import plot_and_compute_metrics
import shutil
import pandas
import numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon
from itertools import product


# -------- before anything else, make the confusion matrices and calculate model accuracy --------
def plot_confusion_matrices():
    path_to_interences = Path("results", "inference")
    run_names = os.listdir(path_to_interences)
    run_names = [name for name in run_names if os.path.isdir(path_to_interences / name)]

    for name in run_names:
        path_json = Path("results", "inference", name, f"{name}.json")
        json_data = json.load(open(path_json))
        plot_and_compute_metrics(
            path_json,
            json_data,
            save_to=Path("results", "confusion_matrices", name, f"{name}.png"),
            text_save_to=Path("results", "confusion_matrices", name, f"{name}.json"),
            accuracy_type="unbalanced",  # "balanced" or "unbalanced"
        )


plot_confusion_matrices()

# -------- now rest of the code depends on the stats by above--------
run_names = Path("results", "inference")
run_names = os.listdir(run_names)

path_registry = Path("results", "inference_registry.json")
registry = json.load(open(path_registry))


LESION_LABELS = {
    0: "0 - No Lesions",
    1: "1 - Granulomas, less than 5",
    2: "2 - Granulomas, more than 5",
    3: "3 - Hydrocephalus",
    4: "4 - Hyperintensities",
    5: "5 - Multiple lesions, other",
}


def get_hyperparams(run_name):
    return registry[run_name]


# -------- metric --------
def get_metric_from_run(run_name, metric="Accuracy"):
    path_json = Path("results", "confusion_matrices", run_name, f"{run_name}.json")
    results = json.load(open(path_json))
    return results[metric]


# -------- data source --------
def get_data_source_from_run(run_name):
    return get_hyperparams(run_name)["meta_input_mode"]


# -------- loops --------
def get_best_run(run_names, metric="Accuracy"):
    runs = run_names
    runs = [(run, get_metric_from_run(run, metric)) for run in runs]
    best_run = max(runs, key=lambda x: x[1])
    print(f"best run: {best_run}")
    return best_run[0]


def plot_metrices_violin(run_names, metric="Accuracy", label="Lesion Type", baseline=0.16666):
    """Plot data source, imaging, clinical, and both using a violin plot"""
    # -------- getting metrics and data sources --------
    values = [get_metric_from_run(run, metric) for run in run_names]
    data_sources = [get_data_source_from_run(run) for run in run_names]

    # -------- make a DataFrame for plotting --------
    df = pandas.DataFrame({"Values": values, "DataSource": data_sources})

    # -------- defining the order for the x-axis --------
    df["DataSource"] = pandas.Categorical(df["DataSource"], categories=["BOTH", "ONLY_IMAGE", "ONLY_CLINICAL"], ordered=True)

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="DataSource", y="Values", inner="quartile", palette="pastel")
    sns.stripplot(data=df, x="DataSource", y="Values", size=8, jitter=False, color="black")

    plt.axhline(y=baseline, color="r", linestyle="--")
    adjust = 0.04 if label == "Lesion Type" else 0
    plt.annotate(
        "Baseline",
        xy=(0, baseline + adjust),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(-20, 5),
        ha="left",
        color="r",
        fontsize=14,
    )
    plt.title(f"Distribution of {metric} in the Test Set when {label} is Used as the Label")
    plt.ylabel(metric.capitalize(), fontsize=16)
    if label == "Lesion Type":
        plt.ylim(0, 0.6)
    else:
        plt.ylim(0, 1)
    plt.xlabel("Data Source")
    plt.tight_layout()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(Path("results", "images", "accuracy", f"{metric.lower()}_{label.lower()}.png"), dpi=420)


class CompareTopNPerformanceAcrossGroups:
    def __init__(self, metric="accuracy", label_type="TBM", modality="ALL"):
        """
        # metric: f1_score / accuracy
        # label_type: # TBM / LESION_TYPE
        # modality: # ONLY_IMAGE / ONLY_CLINICAL / BOTH / ALL
        """
        # -------- primary settings --------
        self.save_to = Path("results", "images", "report", "hiv_vs_non_hiv")
        self.save_to.mkdir(exist_ok=True, parents=True)
        self.label_type = label_type
        self.metric = metric
        self.modality = modality
        self.top_n = 10

        # -------- secondary settings --------
        runs = [run for run in run_names if get_hyperparams(run)["meta_use_label"] == self.label_type]
        runs = [(run, get_metric_from_run(run, "Accuracy")) for run in runs]  # sort by accuracy
        runs = sorted(runs, key=lambda x: x[1], reverse=True)
        self.runs = runs
        if self.top_n:
            self.runs = self.runs[: self.top_n]

        # -------- make placeholders --------
        self.save_text = ""
        self.load_jsons()  # label and metric general
        self.compute_correctness()  # label and metric general

        # -------- make data --------
        self.make_data_lesion_type()  # label specific; metric specific
        self.make_plot()  # done (label and metric specific)
        self.make_compare_stats()

    def load_jsons(self):
        json_lists = []
        for run in self.runs:
            with open(Path("results", "inference", run[0], f"{run[0]}.json"), "r") as loader:
                run_data = json.load(loader)
                for d in run_data:
                    d["run_name"] = run[0]  # run is a ruple (name, accuracy)
                    d["modality"] = get_hyperparams(run[0])["meta_input_mode"]
                json_lists += run_data
        self.json_list = json_lists
        if self.label_type == "TBM":  # correction:
            for i in self.json_list:
                i["label"] += 1
                i["prediction"] += 1

    def compute_correctness(self):
        for i in self.json_list:
            i["prediction_correct"] = i["prediction"] == i["label"]

    # -------- scoring related --------
    def get_model_category_accuracy(self, run_name, label, hiv_status):
        """getting accuracy using model and lesion type"""
        # -------- filder runs by conditions --------
        subset = [i for i in self.json_list if i["run_name"] == run_name]
        subset = [i for i in subset if i["label"] == label]
        subset = [i for i in subset if i["hiv_status"] == hiv_status]
        if self.modality != "ALL":
            subset = [i for i in subset if i["modality"] == self.modality]
        if len(subset) == 0:
            return 0

        # -------- actual computation --------
        trails_totol = len(subset)
        trails_correct = len([i for i in subset if i["prediction_correct"]])
        return trails_correct / trails_totol

    def get_model_category_f1_score(self, run_name, label, hiv_status):
        # -------- filder runs by conditions --------
        subset = [i for i in self.json_list if i["run_name"] == run_name]
        subset = [i for i in subset if i["label"] == label]
        subset = [i for i in subset if i["hiv_status"] == hiv_status]
        if self.modality != "ALL":
            subset = [i for i in subset if i["modality"] == self.modality]

        # -------- actual computation --------
        tp = len([i for i in subset if i["label"] == label == i["prediction"]])
        fn = len([i for i in subset if i["label"] == label and i["prediction"] != i["label"]])
        fp = len([i for i in subset if i["label"] != label and i["prediction"] == label])

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        return f1_score

    # -------- meta utils --------
    def make_plot(self):
        sns.set_context("talk")
        plt.figure(figsize=(8, 8))
        sns.violinplot(
            data=self.data,
            x=self.label_type.lower(),
            y=self.metric,
            hue="hiv_status",
            hue_order=["HIV-", "HIV+"],
            split=True,
            inner="quartiles",
            bw=0.6,  # Adjust the bandwidth to control smoothing
        )
        plt.title(
            f"Distribution of {self.metric.capitalize()} by {self.label_type.capitalize()} in the Top {self.top_n} Models Ranked by Accuracy",
            fontsize=12,
        )
        plt.xlabel("")  # Removes the x-axis label
        # plt.ylabel(self.metric.capitalize(), fontsize=16)
        plt.legend(title="HIV Status")
        plt.xticks(rotation=45, ha="right", fontsize=12)  # Rotate x-axis labels for better visibility
        plt.tight_layout()
        # plt.ylim(0, 1) # Ensures that the y-axis is within [0, 1]
        plt.savefig(self.save_to / f"top_{self.top_n}_{self.label_type}_{self.modality}_hiv_groups_{self.metric}.png", dpi=420)
        plt.clf()
        plt.close()

    def make_compare_stats(self):
        """compare metric between HIV and non-hiv groups for each of the lesions"""
        lesion_types = self.data[self.label_type.lower()].unique()
        significant_lesions = []

        for lesion in lesion_types:
            metric_hiv = self.data[(self.data[self.label_type.lower()] == lesion) & (self.data["hiv_status"] == "HIV+")][
                self.metric
            ].values
            metric_non_hiv = self.data[(self.data[self.label_type.lower()] == lesion) & (self.data["hiv_status"] == "HIV-")][
                self.metric
            ].values

            # Ensure we have paired data
            if len(metric_hiv) == len(metric_non_hiv) and len(metric_hiv) > 0:
                try:
                    stat, p = wilcoxon(metric_hiv, metric_non_hiv)
                    if p < 0.05:
                        significant_lesions.append((lesion, p))
                except ValueError:
                    pass

        if significant_lesions:
            self.save_text += (
                f"wilcoxon: {self.label_type} with significant differences in {self.metric} between HIV and non-HIV groups:"
            )
            self.save_text += "\n\n"
            for lesion, p in significant_lesions:
                self.save_text += (
                    f"wilcoxon: {self.label_type}: {lesion} has a significant difference in {self.metric} with p-value: {p:.5f}"
                )
        else:
            self.save_text += f"wilcoxon: No {self.label_type} showed a significant difference in {self.metric} between HIV and non-HIV groups."

        with open(
            self.save_to / f"top_{self.top_n}_{self.label_type}_{self.modality}_hiv_groups_{self.metric}.MD", "w"
        ) as writer:
            json.dump(self.save_text, writer, indent=4)

    # -------- lesion type specific --------
    def make_data_lesion_type(self):
        """make the self.dataframe using the parameters"""
        run_names = [i[0] for i in self.runs]
        run_names = set(run_names)
        labels = set([i["label"] for i in self.json_list])
        hiv_status = ["HIV-", "HIV+"]

        self.data = []
        for name, label, status in list(product(run_names, labels, hiv_status)):
            if self.label_type == "LESION_TYPE":
                label_name = LESION_LABELS[label]
            else:
                label_name = label
            self.data.append(
                {
                    "run_name": name,
                    f"{self.label_type.lower()}": label_name,
                    "hiv_status": status,
                    "accuracy": self.get_model_category_accuracy(name, label, status),
                    "f1_score": self.get_model_category_f1_score(name, label, status),
                }
            )

        self.data = pandas.DataFrame(self.data)


if __name__ == "__main__":
    # -------- plot accuracy across models --------
    def plot_accuracy_lesion_type():
        runs = run_names  # make local
        runs = [run for run in runs if get_hyperparams(run)["meta_use_label"] == "LESION_TYPE"]
        plot_metrices_violin(runs, metric="Accuracy", label="Lesion Type", baseline=0.16666)

    def plot_accuracy_tbm():
        runs = run_names  # make local
        runs = [run for run in runs if get_hyperparams(run)["meta_use_label"] == "TBM"]
        plot_metrices_violin(runs, metric="Accuracy", label="TBM", baseline=0.33333)

    plot_accuracy_lesion_type()
    plot_accuracy_tbm()

    # -------- get confusion matrics of the best runs --------
    def get_top_n_confusion_matrices_lesion_type(n=3):
        save_to = Path("results", "images", "report")
        runs = run_names
        runs = [run for run in runs if get_hyperparams(run)["meta_use_label"] == "LESION_TYPE"]
        runs = [(run, get_metric_from_run(run, "Accuracy")) for run in runs]
        runs = sorted(runs, key=lambda x: x[1], reverse=True)
        runs = runs[:n]
        print(runs)
        for run in runs:
            path_report_data = Path("results", "confusion_matrices", run[0], f"{run[0]}.json")
            path_confusion_matrix = Path("results", "confusion_matrices", run[0], f"{run[0]}.png")
            new_name = f"lesion_type_top_{runs.index(run) + 1}_{run[0]}_accuracy_{run[1]:.4f}"
            shutil.copy(path_confusion_matrix, save_to / (new_name + ".png"))
            shutil.copy(path_report_data, save_to / (new_name + ".json"))

    def get_top_n_confusion_matrices_tbm(n=3):
        save_to = Path("results", "images", "report")
        runs = run_names
        runs = [run for run in runs if get_hyperparams(run)["meta_use_label"] == "TBM"]
        runs = [(run, get_metric_from_run(run, "Accuracy")) for run in runs]
        runs = sorted(runs, key=lambda x: x[1], reverse=True)
        runs = runs[:n]
        for run in runs:
            path_report_data = Path("results", "confusion_matrices", run[0], f"{run[0]}.json")
            path_confusion_matrix = Path("results", "confusion_matrices", run[0], f"{run[0]}.png")
            new_name = f"tbm_top_{runs.index(run) + 1}_{run[0]}_accuracy_{run[1]:.4f}"
            shutil.copy(path_confusion_matrix, save_to / (new_name + ".png"))
            shutil.copy(path_report_data, save_to / (new_name + ".json"))

    get_top_n_confusion_matrices_lesion_type()
    get_top_n_confusion_matrices_tbm()

    # -------- compare the models' performance on hiv+ vs hiv- groups --------
    def get_top_n_model_performance_lesion_type_hiv(n=3, hive_status="HIV-"):
        save_to = Path("results", "images", "report")
        runs = run_names
        runs = [run for run in runs if get_hyperparams(run)["meta_use_label"] == "LESION_TYPE"]
        runs = [(run, get_metric_from_run(run, "Accuracy")) for run in runs]
        runs = sorted(runs, key=lambda x: x[1], reverse=True)
        runs = runs[:n]
        for run in runs:
            path_json = Path("results", "inference", run[0], f"{run[0]}.json")

            with open(path_json, "r") as loader:
                json_data = json.load(loader)

            json_data = [data for data in json_data if data["hiv_status"] == hive_status]

            plot_and_compute_metrics(
                path_json=path_json,
                json_data=json_data,
                save_to=save_to / f"hiv_lesion_type_top_{runs.index(run) + 1}_{hive_status}_{run[0]}.png",
                text_save_to=save_to / f"hiv_lesion_type_top_{runs.index(run) + 1}_{hive_status}_{run[0]}.json",
            )

    get_top_n_model_performance_lesion_type_hiv(n=3, hive_status="HIV-")
    get_top_n_model_performance_lesion_type_hiv(n=3, hive_status="HIV+")

    instance = CompareTopNPerformanceAcrossGroups(metric="accuracy", label_type="LESION_TYPE", modality="ALL")
    instance = CompareTopNPerformanceAcrossGroups(metric="f1_score", label_type="LESION_TYPE", modality="ALL")
    instance = CompareTopNPerformanceAcrossGroups(metric="accuracy", label_type="TBM", modality="ALL")
    instance = CompareTopNPerformanceAcrossGroups(metric="f1_score", label_type="TBM", modality="ALL")

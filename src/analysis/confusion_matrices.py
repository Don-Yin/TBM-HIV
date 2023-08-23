import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score


def get_label_type(path):
    # read the registry
    path_inference_registry = Path("results", "inference_registry.json")
    with path_inference_registry.open("r") as file:
        registry = json.load(file)

    # get the config using name
    config = registry[path.stem]
    return config["meta_use_label"]


def plot_and_compute_metrics(
    path_json,
    json_data,
    save_to=Path("test.png"),
    text_save_to=Path("test.json"),
):
    true_labels = [entry["label"] for entry in json_data]
    predictions = [entry["prediction"] for entry in json_data]

    # -------- label_type --------
    label_type = get_label_type(path_json)

    if label_type == "TBM":
        true_labels = [i + 1 for i in true_labels]
        predictions = [i + 1 for i in predictions]

    classes = sorted(list(set(true_labels)))
    acc = accuracy_score(true_labels, predictions)
    sensitivity = recall_score(true_labels, predictions, average=None, labels=classes)
    f1 = f1_score(true_labels, predictions, average=None, labels=classes)

    text_result = {}
    print(f"Accuracy: {acc}")
    text_result.update({"Accuracy": acc})
    for idx, c in enumerate(classes):
        print(f"Class {c}: Sensitivity (Recall): {sensitivity[idx]}, F1-score: {f1[idx]}")
        text_result.update({f"Class {c}": {"Sensitivity": sensitivity[idx], "F1-score": f1[idx]}})

    # Step 4: Plot confusion matrix
    sns.set_context("talk")
    cm = confusion_matrix(true_labels, predictions, labels=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("True", fontsize=18)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(save_to, dpi=420)

    with text_save_to.open("w") as file:
        json.dump(text_result, file, indent=4)


if __name__ == "__main__":
    path_json = Path("results", "inference", "test_lesion.json")
    plot_and_compute_metrics(path_json)

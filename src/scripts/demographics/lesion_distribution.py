import os
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from src.imaging.modules.scan import Scan

# if this fails refer to the image registry at cache
path_images = Path("/Users/donyin/Desktop/images_only_first")
path_label = Path("data", "MRI_sessions.csv")
frame_label = pd.read_csv(path_label)
folders = os.listdir(path_images)
folders = [i for i in folders if not i.startswith(".")]

labels = {
    0: "0 - No Lesions",
    1: "1 - Granulomas, less than 5",
    2: "2 - Granulomas, more than 5",
    3: "3 - Hydrocephalus",
    4: "4 - Hyperintensities",
    5: "5 - Multiple lesions, other",
}


def plot_lesion_distribution(path):
    images = os.listdir(path)
    scans = [Scan(i) for i in images]
    lesion_type = [i.lesion_type for i in scans]
    lesion_type = Counter(lesion_type)
    lesion_type = dict(natsorted(lesion_type.items(), key=lambda item: item[0]))

    data_type = path.stem.capitalize()
    df = pd.DataFrame(
        {"lesion_type": list(lesion_type.keys()), "count": list(lesion_type.values()), "data": [data_type] * len(lesion_type)}
    )

    return df


def create_bar_chart(plot_data):
    plt.figure(figsize=(7, 7))
    barplot = sns.barplot(data=plot_data, x="lesion_type", y="count", hue="data")

    plt.title("Distribution of Lesion Types across Datasets", fontsize=16)
    plt.xlabel("Lesion Type", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(ticks=range(len(labels)), labels=labels.values(), rotation=45)

    ymax = plot_data["count"].max() * 1.10
    plt.ylim(0, ymax)

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

    save_to = Path("results", "images", "demographics", "lesion_distribution.png")
    save_to.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_to, dpi=420)


if __name__ == "__main__":
    all_data = []

    for folder in folders:
        path = path_images / folder
        df = plot_lesion_distribution(path)
        all_data.append(df)

    plot_data = pd.concat(all_data, ignore_index=True)
    create_bar_chart(plot_data)

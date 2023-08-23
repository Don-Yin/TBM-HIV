"""
this script keeps the first scan (by date) of each subject and filters out that do not have a label in MR_Sessions.csv; from 302 to 255 images
"""


import os
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
from src.imaging.modules.scan import Scan

# if this fails refer to the image registry at cache
path_images = Path("/Users/donyin/Desktop/images_only_first")
path_label = Path("data", "MRI_sessions.csv")
frame_label = pd.read_csv(path_label)

folders = os.listdir(path_images)
folders = [i for i in folders if not i.startswith(".")]


def clear_folder(path):
    images = os.listdir(path)
    scans = [Scan(i) for i in images]

    # Group scans by subject_id
    scan_groups = {}
    for scan in scans:
        if scan.subject_id not in scan_groups:
            scan_groups[scan.subject_id] = []
        scan_groups[scan.subject_id].append(scan)

    # For each group, sort by date and keep the earliest
    for _, scan_list in scan_groups.items():
        scan_list.sort(key=lambda s: s.date)
        # Remove scans other than the earliest one
        for scan in scan_list[1:]:
            os.remove(path / scan.file_name)
            print(f"removing {scan.file_name}")


def check_distribution(path):
    """checking the distribution of the number of lesion type in a folder"""
    images = os.listdir(path)
    scans = [Scan(i) for i in images]

    # count lesion type in the folder
    from collections import Counter

    lesion_type = [i.lesion_type for i in scans]
    lesion_type = Counter(lesion_type)

    # sort the counter by key
    lesion_type = dict(natsorted(lesion_type.items(), key=lambda item: item[0]))

    # remove the nan files
    for scan in scans:
        if np.isnan(scan.lesion_type):
            os.remove(path / scan.file_name)
            print(f"removing {scan.file_name}")


if __name__ == "__main__":
    pass
    # for folder in folders:
    #     clear_folder(path_images / folder)

    # for folder in folders:
    #     check_distribution(path_images / folder)

"""
IMPORTANT: 
This script assumes that:
    - the train, validation and test subjects are splitted with their IDs stored in the cache/split.json file
    - all image volumes (to be splitted) are stored in a single folder, named as "IMAGE_FOLDER" (can be adjusted in this script)
    - there is a label.csv file that contains two columns: subject_id and MR

It then splits the images into train, validation and test sets, each stored in a separate folder.    
"""

import os
from pathlib import Path
import pandas
import json
import shutil
from src.utils.general import banner
from tqdm import tqdm

# -------- set the paths --------
# change this to the path of the image folderƒ
PATH_IMAGE_FOLDER = Path("IMAGE_FOLDER")
# change this to the path of the label.csv that contains the subject_id and MR columns
PATH_LABEL_CSV = Path("data", "MRI_sessions.csv")

# -------- set target paths --------
move_path = Path("data", "images")
target_paths = {i: move_path / i for i in ["train", "valid", "test"]}
# make sure the target paths exist
for i in target_paths.values():
    os.makedirs(i, exist_ok=True)

# -------- read the json file --------
with open(Path("cache", "split.json"), "r") as loader:
    split_dict = json.load(loader)

# -------- read the label.csv and the images --------
label_df = pandas.read_csv(PATH_LABEL_CSV)
images = os.listdir(PATH_IMAGE_FOLDER)
images = [i for i in images if i.endswith(".nii")]


def mr_to_subject_id(mr_id):
    try:
        return label_df[label_df["MR ID"] == mr_id]["subject_id"].values[0]
    except IndexError:
        return None


def file_name_to_mr_id(name):
    return "_".join(name.split("_")[:2])


def file_name_to_subject_id(name):
    mr_id = file_name_to_mr_id(name)
    return mr_to_subject_id(mr_id)


def file_name_to_category(name):
    subject_id = file_name_to_subject_id(name)
    # find which key the subject_id belongs to
    for key in split_dict.keys():
        if subject_id in split_dict[key]:
            return key


banner(f"Copying images to {str(move_path)}")

counter = {"train": 0, "valid": 0, "test": 0}
progress_bar = tqdm(total=len(images))

unknown_participants = []
unknown_mr_id = []

for i in images:
    mr_id = file_name_to_mr_id(i)
    subject_id = file_name_to_subject_id(i)
    category = file_name_to_category(i)

    if not subject_id:
        unknown_mr_id.append(mr_id)
        continue

    if not category:
        unknown_participants.append(subject_id)
        continue

    counter[category] += 1
    progress_bar.update(1)
    progress_bar.set_description(f"Moving {i} to {category}")
    shutil.copy(Path(PATH_IMAGE_FOLDER, i), Path(target_paths[category], i))

# BMEIS_S19260
print(f"Found {len(unknown_participants)} unknown participants: {unknown_participants}\n")
print(f"Unknown MR ID: {unknown_mr_id}\n")

progress_bar.close()

banner(f"Number of images in each category: {counter}")

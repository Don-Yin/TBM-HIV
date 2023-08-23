from src.imaging.utils.transformations import CustomRand3DElastic, CustomRandAffine, CustomRandGaussianNoise
from src.imaging.utils.clinical_data import get_clinical_data
from torch.utils.data import Dataset
from src.utils.general import banner
from collections import Counter
import matplotlib.pyplot as plt
import monai.transforms as mt
from pathlib import Path
from rich import print
import numpy as np
import pandas
import torch


class NiftiDataset(Dataset):
    def __init__(self, images_path, use_label="TBM", augment=False):
        self.label_frame = pandas.read_csv(Path("data", "MRI_sessions.csv"))
        self.clinical_frame = pandas.read_csv(Path("data", "1_clinical_merged", "clinical.csv"))

        self.image_files = sorted([f for f in images_path.glob("*.nii")])
        self.augment = augment
        self.use_label = use_label

        affine_transform = CustomRandAffine(
            keys=["image"],
            prob=0.15,
            rotate_range=(np.pi / 6, np.pi / 6, np.pi / 6),  # Up to 30 degrees each axis.
            shear_range=(0.2, 0.2, 0.2),  # Increased shear effect.
            translate_range=(10, 10, 10),  # Translations (in pixel units) increased.
            scale_range=(0.2, 0.2, 0.2),  # Increased scaling.
            padding_mode="reflection",  # Options: 'constant', 'edge', 'symmetric', 'reflect', 'wrap'
        )

        elastic_transform = CustomRand3DElastic(
            keys=["image"],
            prob=0.2,
            sigma_range=(1, 2),
            magnitude_range=(0, 0.2),
            spatial_size=None,
            padding_mode="reflection",
        )

        gaussian_noise = CustomRandGaussianNoise(
            keys=["image"],
            prob=0.2,
            mean=0.0,
            std=0.2,
        )

        self.transform = mt.Compose(
            [
                mt.LoadImaged(keys=["image"], image_only=True),
                mt.EnsureChannelFirstd(keys=["image"]),
                *([affine_transform, elastic_transform, gaussian_noise] if augment else []),
                mt.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                mt.ToTensord(keys=["image"]),
            ]
        )

        self.max_dimensions = self.get_max_dimensions()

        if self.augment:
            banner("Viewing the augmentation results")
            [self.view_augmentation_example(i) for i in range(3)]

    # -------- configurable --------
    def idx_label_function(self, idx):
        if self.use_label == "TBM":
            return self.idx_to_label_tbm_grade(idx)
        elif self.use_label == "LESION_TYPE":
            return self.idx_to_label_lesion_type(idx)

    # -------- dunder --------
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):  # change this to __getitem__ to get the entire image / get_entire_image
        data = {"image": str(self.image_files[idx])}
        image = self.transform(data)["image"]
        image = self.pad_to_max_dimensions(image)
        return image, self.idx_label_function(idx), self.idx_to_clinical_data(idx)

    # -------- idx functions --------
    def idx_to_item_no_padding(self, idx):
        data = {"image": str(self.image_files[idx])}
        image = self.transform(data)["image"]
        return image, self.idx_label_function(idx), self.idx_to_clinical_data(idx)

    def idx_to_clinical_data(self, idx):
        img_name = self.image_files[idx].name
        mr_id = img_name.split("_")[0] + "_" + img_name.split("_")[1]
        subject_id = self.id_conversion(mr_id, "subject")
        clinical_data = get_clinical_data(subject_id)
        return clinical_data

    def idx_to_label_lesion_type(self, idx):
        """
        A very ugly function to get the image label using the image file name
        """
        img_name = self.image_files[idx].name
        mr_id = img_name.split("_")[0] + "_" + img_name.split("_")[1]
        row = self.label_frame.loc[self.label_frame["MR ID"] == mr_id]
        label_str = row["Type"].values[0]  # e.g., '2 - Granulomas, more than 5' / nan

        # # ---------- [ dealing with nan values ] ----------
        # try:
        #     assert not pandas.isnull(label_str), f"label_str is nan for MR ID {mr_id}"
        #     label_value = int(label_str[0])  # e.g., 2
        # except AssertionError:
        #     label_value = 0  # 0 for no lesion (for now)
        # # -------------------------------------------------

        assert not pandas.isnull(label_str), f"label_str is nan for MR ID {mr_id}"
        label_value = int(label_str[0])  # e.g., 2

        return label_value

    def idx_to_subject_id(self, idx):
        img_name = self.image_files[idx].name
        mr_id = img_name.split("_")[0] + "_" + img_name.split("_")[1]
        subject_id = self.id_conversion(mr_id, "subject")
        return subject_id

    def idx_to_project(self, idx):
        subject_id = self.idx_to_subject_id(idx)
        row = self.clinical_frame.loc[self.clinical_frame["subject_id"] == subject_id]
        project = row["project_bidhae"].values[0]
        assert project in ["OUCRU_TBM27", "OUCRU_TBM26"], f"project is {project}"
        return project

    def idx_to_hiv_status(self, idx):
        project = self.idx_to_project(idx)
        return "HIV+" if project == "OUCRU_TBM26" else "HIV-"

    def idx_to_label_tbm_grade(self, idx):
        img_name = self.image_files[idx].name
        mr_id = img_name.split("_")[0] + "_" + img_name.split("_")[1]
        subject_id = self.id_conversion(mr_id, "subject")
        row_clinical = self.clinical_frame.loc[self.clinical_frame["subject_id"] == subject_id]
        tbm_grade = row_clinical["tbmgrade_followupsigns"].values[0]
        return int(tbm_grade) - 1

    # ---------- [ helper functions ] ----------
    def pad_to_max_dimensions(self, image: torch.Tensor):
        """
        Pad the image to the maximum dimensions if its dimensions are smaller
        """
        spatial_size = self.max_dimensions.tolist()
        spatial_pad = mt.SpatialPadd(keys=["image"], spatial_size=spatial_size, mode="constant")
        image = spatial_pad({"image": image})["image"]
        return image

    def get_max_dimensions(self):
        """
        getting the max dimensions from different axis of all images in the current dataset
        """
        xs = [self.idx_to_item_no_padding(i)[0].shape[1] for i in range(len(self))]
        ys = [self.idx_to_item_no_padding(i)[0].shape[2] for i in range(len(self))]
        zx = [self.idx_to_item_no_padding(i)[0].shape[3] for i in range(len(self))]

        max_dimensions = torch.tensor((max(xs), max(ys), max(zx)))
        return max_dimensions

    def get_classes(self):
        """
        getting the classes of all images in the current dataset from the label
        """
        return set([self.idx_label_function(i) for i in range(len(self))])

    def get_num_samples_per_class(self):
        """
        Counts the number of samples per class in the dataset and return it as a dictionary
        """
        classes = [self.idx_label_function(i) for i in range(len(self))]
        class_counts = Counter(classes)
        return dict(class_counts)

    def id_conversion(self, num, target_system: int):
        """
        Convert the image id to the other system
        """
        if target_system == "subject":
            row = self.label_frame.loc[self.label_frame["MR ID"] == num]
            label = row["subject_id"].values[0]
        elif target_system == "mr":
            row = self.label_frame.loc[self.label_frame["subject_id"] == num]
            label = row["MR ID"].values[0]

        return label

    def view_augmentation_example(self, idx, num_slices=12):
        """visualize side by side comparison of images with and without augmentation"""

        assert self.augment is True, "Augment boolean has to be True to use this method"
        num_slices += 2  # for later removing index 0 and -1
        save_to = Path("results", "images", f"augmentation_{idx}.png")

        transform_no_augment = mt.Compose(
            [
                mt.LoadImaged(keys=["image"], image_only=True),
                mt.EnsureChannelFirstd(keys=["image"]),
                mt.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                mt.ToTensord(keys=["image"]),
            ]
        )

        data = {"image": str(self.image_files[idx])}

        data_original = transform_no_augment(data)
        data_augment = self.transform(data)

        image_original = data_original["image"].numpy()[0]  # Channel First, so select first channel
        image_augment = data_augment["image"].numpy()[0]
        slice_indices = np.linspace(0, image_original.shape[0] - 1, num_slices, dtype=int)

        # comback to the intended number after the indeces calculation
        num_slices -= 2
        _, axes = plt.subplots(num_slices, 2, figsize=(12, 6 * num_slices))

        for i, slice_idx in enumerate(slice_indices[1:-1]):
            axes[i, 0].imshow(image_original[slice_idx], cmap="gray")
            axes[i, 0].set_title(f"Original Image, Slice {slice_idx}")

            axes[i, 1].imshow(image_augment[slice_idx], cmap="gray")
            axes[i, 1].set_title(f"Augmented Image, Slice {slice_idx}")

        plt.tight_layout()
        plt.savefig(save_to)
        print(f"Examples of image {idx} saved in {str(save_to)}")


if __name__ == "__main__":
    dataset_valid = NiftiDataset(images_path=Path("data", "IMAGE_FOLDER"), augment=False)
    for image, label, clinical_data in dataset_valid:
        print(label)

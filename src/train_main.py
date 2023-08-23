"""
The main script for training the image model in a classification tasks
"""

from src.imaging.utils.general import device, apply_kaiming_initialization
from monai.networks.layers.factories import Act, Norm
from src.imaging.utils.training import main_training_loop
from src.imaging.utils.dataset import NiftiDataset
from src.imaging.utils.models import CombinedModel
from src.imaging.utils.logger import get_logger
from src.utils.general import banner
import torch.optim as optim
from pathlib import Path
from rich import print
from torch import nn
import wandb


import torch


def main(config, epochs, epochs_min, logger_name="tensorboard"):
    banner(f"Using device: {device}")
    banner("Starting with Configurations")
    print(config)

    logger_handler = get_logger(logger_name)
    context = logger_handler.get_context(config, project="DenseNet")  # project only applies to wandb

    with context:
        # -------- setting up the dataset ---------
        dataset_train = NiftiDataset(
            # images_path=Path("/jmain02/home/J2AD019/exk01/ddy19-exk01/images/train"),
            images_path=Path("/Users/donyin/Desktop/images_only_first/train"),  # if on server amend
            use_label=config["meta_use_label"],
            augment=config["meta_augment"],
        )

        dataset_valid = NiftiDataset(
            # images_path=Path("/jmain02/home/J2AD019/exk01/ddy19-exk01/images/valid"),
            images_path=Path("/Users/donyin/Desktop/images_only_first/valid"),
            use_label=config["meta_use_label"],
            augment=False,
        )

        # ------ check the train and test data both have all the possible labels ------
        classes_train = dataset_train.get_classes()
        classes_valid = dataset_valid.get_classes()

        error_message = f"The train and validation label classes have to be the same\nCurrently, the train classes are: {classes_train}\nwhile the validation classes are: {classes_valid}"
        assert classes_valid.issubset(classes_train), error_message

        # ------ unify the max dimensions of the train and test data ------------------
        max_dimensions_overall = torch.max(dataset_train.max_dimensions, dataset_valid.max_dimensions)
        dataset_train.max_dimensions = max_dimensions_overall
        dataset_valid.max_dimensions = max_dimensions_overall

        # ------ from here the __getitem__ function will pad the images ---------------
        output_classes = len(classes_train)
        banner(f"Number of classes detected in the train dataset: {output_classes}; proceeding...")

        # ------ instantiate the model -----------------------------------------------
        model_combined = CombinedModel(config).to(device)

        # ------ checkpoint path -----------------------------------------------------
        checkpoint_path = Path("checkpoints", logger_handler.get_run_name())  # for saving checkpoints
        apply_kaiming_initialization(model_combined)

        # Training ---------------------------------------------------------------
        optimizer = optim.Adam(model_combined.parameters(), lr=config["meta_learning_rate"])

        # ---------------- setting up the weighted loss function ----------------
        if config["meta_use_weighted_loss"]:
            class_counts = dataset_train.get_num_samples_per_class()  # {2: 1, 4: 3, 1: 1, 3: 3, 5: 3}
            class_weights = [1 / class_counts[i] for i in range(len(class_counts))]
            class_weights = [weight / sum(class_weights) for weight in class_weights]
            class_weights_tensor = torch.tensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        # ---------------- setting up the logger ----------------
        logger_handler.handle_watch_model(model_combined, criterion)
        logger_handler.log_hyperparameters(config | {"image_dimensions": max_dimensions_overall.tolist()})
        logger_handler.log_init_images()

        main_training_loop(
            model_combined=model_combined,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            dataset_train=dataset_train,
            dataset_valid=dataset_valid,
            checkpoint_path=checkpoint_path,
            batch_size=config["meta_batch_size"],
            logger_handler=logger_handler,
            # early stopping epochs:
            epochs_min=epochs_min,
        )


if __name__ == "__main__":
    pass

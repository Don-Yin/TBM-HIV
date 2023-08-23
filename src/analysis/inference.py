"""
accuracy testing on test set
sensitivity analysis

"""
from src.analysis.confusion_matrices import plot_and_compute_metrics
from src.imaging.utils.models import CombinedModel
from src.imaging.utils.general import load_checkpoint
from src.imaging.utils.dataset import NiftiDataset
from src.imaging.utils.general import device
from monai.networks.layers.factories import Act
from pathlib import Path
from tqdm import tqdm
import torch
import json


def inference(model, image, clinical_data):
    logits = model(image, clinical_data)
    _, predicted = torch.max(logits, 1)
    return predicted, logits


def save_to_registry(config, save_as):
    path_registry = Path("results", "inference_registry.json")
    path_registry.parent.mkdir(parents=True, exist_ok=True)

    # if the registry file doesn't exist, create it
    if not path_registry.exists():
        json.dump({}, open(path_registry, "w"))

    # load the registry file
    registry = json.load(open(path_registry, "r"))
    registry.update({save_as.stem: config})

    # save the registry file
    json.dump(registry, open(path_registry, "w"))


def inference_ensemble(
    config,
    model_combined,
    path_images_folder,
    save_as,
    path_checkpoint=None,
):
    """
    path_images_folder: Path to the folder containing the images for inference
    path_labels: Path to the csv file containing the labels for inference; this associate images with clinical data
    path_save_to: the folder where to result json file goes
    """
    save_as.parent.mkdir(parents=True, exist_ok=True)

    model_combined.to(device).eval()
    if path_checkpoint:
        load_checkpoint(checkpoint_path=path_checkpoint, model=model_combined)

    dataset = NiftiDataset(images_path=path_images_folder, use_label=config["meta_use_label"], augment=False)

    results = []

    for i in tqdm(range(len(dataset))):
        image, label, clinical_data = dataset[i]
        image, label, clinical_data = image.to(device), label, clinical_data.to(device)
        image = image.unsqueeze(0)
        clinical_data = clinical_data.unsqueeze(0)
        predicted, logits = inference(model=model_combined, image=image, clinical_data=clinical_data)

        logits = logits.detach().cpu().numpy().squeeze().tolist()

        results.append(
            {
                "image_name": dataset.image_files[i].name,
                "subject_id": dataset.idx_to_subject_id(i),
                "project": dataset.idx_to_project(i),
                "hiv_status": dataset.idx_to_hiv_status(i),
                "prediction": predicted.item(),
                "label": label,
                "logits": logits,
            }
        )

    json.dump(results, open(save_as, "w"))
    save_to_registry(config=config, save_as=save_as)


if __name__ == "__main__":
    # ---------------- loading the model ----------------
    configurations = {
        # -------- mlp: fix this for now --------
        "mlp_input_size": [53],  # 56 with full varibles and 53 for without gcs
        "mlp_output_size": [8],
        "mlp_hidden_sizes": [[32, 24, 16]],
        "mlp_dropout_prob": [0.05],
        "mlp_activation_function": [Act.LEAKYRELU],
        # ------ hyperparameters ------
        "densenet_spatial_dims": [3],
        "densenet_in_channels": [1],
        "densenet_out_channels": [128],  # 32  # final latent size
        "densenet_init_features": [64],
        # default 32; i.e., number of channels at each layer in the block if no dense connections
        "densenet_growth_rate": [32],
        # (6, 12, 32, 32), (6, 12, 48, 32) / (6, 12, 64, 48), 121, 169, 201, 264; it looks like the size doesn't matter when we use the tbm grade as the label; but i'd still like to try different sizes when lesion type is the label
        "densenet_block_config": [(6, 12, 24, 16)],
        "densenet_bn_size": [4],  # bottleneck; channel limit = bn_size * growth_rate
        "densenet_act": [Act.LEAKYRELU],  # "relu"
        "densenet_norm": ["batch"],
        # looks like dropout doesn't matter too much; 0.05, 0.15
        "densenet_dropout_prob": [0.15],
        # -------- overall --------
        "meta_batch_size": [4],
        "meta_learning_rate": [3e-4, 6e-4],  # current best larger then 3e-4
        # has to keep all; ["BOTH", "ONLY_IMAGE", "ONLY_CLINICAL"]
        "meta_input_mode": ["BOTH", "ONLY_IMAGE", "ONLY_CLINICAL"],
        "meta_use_weighted_loss": [False],  # better as false, strange
        "meta_augment": [False, True],  # no effect on the results; the settings are amended, try again
        "meta_combined_model_linear_layer_size": [[64, 32]],
        "meta_use_label": ["LESION_TYPE", "TBM"],  # 3, 6
    }

    # take a random sample from each
    config = {i: configurations[i][0] for i in configurations}
    model_combined = CombinedModel(config=config).to(device)

    inference_ensemble(
        config=config,
        model_combined=model_combined,
        path_images_folder=Path("/Users/donyin/Desktop/images/test"),
        save_as=Path("results", "inference") / "test" / "test.json",
        path_checkpoint=Path("checkpoints", "1972eac9"),  # change this in addition to the model configs
    )
    plot_and_compute_metrics(Path("results", "inference") / "test" / "test.json")

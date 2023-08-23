"""
this is for making inference on the test set and computing the confusion matrices
note:
    this runs on jade, so the paths are different
--------
output:
    results/confusion_matrices
    results/inference
    results/inference_registry.json
"""

import json
import os
from pathlib import Path

from src.imaging.utils.models import CombinedModel
from src.imaging.utils.general import device

from src.analysis.confusion_matrices import plot_and_compute_metrics
from src.analysis.inference import inference_ensemble
from src.imaging.utils.models import CombinedModel

run_path_checkpoints = "/jmain02/home/J2AD019/exk01/ddy19-exk01/densenet-lesion-16-checkpoints"
run_path_configs = "/jmain02/home/J2AD019/exk01/ddy19-exk01/densenet-lesion-16-runs"

run_names = os.listdir(run_path_checkpoints)


def get_config(run_name):
    path_config = Path(run_path_configs) / run_name / "config.json"
    config = json.load(open(path_config))
    return config


for name in run_names:
    config = get_config(name)
    model_combined = CombinedModel(config=config).to(device)
    path_json = Path("results", "inference", name, f"{name}.json")
    path_json.parent.mkdir(parents=True, exist_ok=True)

    inference_ensemble(
        config=config,
        model_combined=model_combined,
        path_images_folder=Path("/Users/donyin/Desktop/images_only_first/test"),  # if on server amend
        save_as=path_json,
        path_checkpoint=Path(run_path_checkpoints) / name,  # change this in addition to the model configs
    )

    plot_and_compute_metrics(
        path_json,
        save_to=Path("results", "confusion_matrices", name, f"{name}.png"),
        text_save_to=Path("results", "confusion_matrices", name, f"{name}.json"),
    )


if __name__ == "__main__":
    pass

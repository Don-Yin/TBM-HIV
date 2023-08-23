"""
Note that the hparams are updated after sycking the tfevents files, so stay patient
Sync the tfevents files from tensorboard to wandb
files have to be in the following structure:
    runs/run_name/tfevents
    runs/run_name/config.json
    ...
"""


from pathlib import Path
import json
import wandb
import os
from rich import print


api = wandb.Api()


def get_run_name(file_str):
    return file_str.split("/")[1]


def sync(path, project):
    """ """
    files = list(Path(path).rglob("*tfevents*"))

    for file in files:
        file = str(file)
        run_name = get_run_name(file)
        os.system(f"wandb sync {file} --id {run_name} --project {project}")

    config_files = list(Path(path).rglob("config.json"))

    for file in config_files:
        file = str(file)
        run_name = get_run_name(file)
        print(f"Found config file for {run_name}, syncing...")
        with open(file, "r") as config_file:
            config = json.load(config_file)

        run = api.run(f"don-yin/{project}/{run_name}")
        run.config.update(config)
        run.update()


if __name__ == "__main__":
    sync(Path("runs_first_image"), "DenseNet-First-Image")

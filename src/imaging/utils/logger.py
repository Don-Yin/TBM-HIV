from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from uuid import uuid4
from rich import print
import wandb
import torch
import json
import os
import numpy as np

from PIL import Image
import torch
from pathlib import Path
import os


def get_logger(name="tensorboard"):
    if name == "tensorboard":
        return TensorboardHandler()
    elif name == "wandb":
        return WandBHandler()
    else:
        return None


class TensorboardHandler:
    def __init__(self):
        self.items = {}
        self.iterations = {}  # Dictionary to keep track of iterations for each key
        self.moving_average = {}
        self.run_name = None
        self.writer = SummaryWriter(log_dir=str(Path("runs", self.get_run_name())))

    # -------- logging functions --------

    def log_hyperparameters(self, config):
        # store to json
        with open(Path("runs", self.get_run_name(), "config.json"), "w") as writer:
            json.dump(config, writer, indent=4)

        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                config[key] = json.dumps(value)  # or str(value)
        self.writer.add_hparams(hparam_dict=config, metric_dict={}, run_name=self.get_run_name())

    def log(self, content: dict):
        """
        content is a dictionary of key-value pairs: e.g., {"train_loss": 0.1}
        """
        self.items.update(content)
        self.moving_average_record(name=next(iter(content.keys())), value=next(iter(content.values())))

        for key in content:
            if key not in self.iterations:
                self.iterations[key] = 0
            self.iterations[key] += 1  # Increment iteration count for the key

            self.writer.add_scalar(key, content[key], self.iterations[key])
            self.writer.flush()

        self.show_in_terminal()

    def log_init_images(self):
        images_path = Path("results", "images")
        files = os.listdir(images_path)
        files = [file for file in files if os.path.isfile(images_path / file)]
        files = [file for file in files if file.endswith(".png")]
        for file in files:
            image_path = images_path / file
            image = Image.open(image_path)
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            self.writer.add_image(file, image_tensor, 0)

    # -------- helper functions --------
    def on_early_stop(self):
        self.writer.flush()
        self.show_in_terminal()

    def on_end(self):
        self.writer.close()
        self.show_in_terminal()

    def handle_watch_model(self, model, criterion):
        pass

    def get_run_name(self):
        if self.run_name:
            return self.run_name
        else:
            self.run_name = str(uuid4().hex[:8])
            return self.run_name

    def get_context(self, config, project):
        return nullcontext()

    # ---------------- show ---------------
    def show_in_terminal(self):
        print("|" + " | ".join([f"{k}: {v.__round__(3)}" for k, v in self.items.items()]) + "|")

    # ---------------- moving average ---------------
    def moving_average_record(self, name, value, window_size=10):
        if name not in self.moving_average:
            self.moving_average[name] = [value]
        else:
            self.moving_average[name].append(value)
            if len(self.moving_average[name]) > window_size:
                self.moving_average[name].pop(0)

    def moving_average_get(self, name):
        return sum(self.moving_average[name]) / len(self.moving_average[name])


class WandBHandler:
    def __init__(self):
        self.items = {}
        self.moving_average = {}

    # -------- logging functions --------

    def log_hyperparameters(self, config):
        wandb.config.update(config)

    def log(self, content):
        self.items.update(content)
        self.moving_average_record(name=next(iter(content.keys())), value=next(iter(content.values())))
        wandb.log(content)
        self.show_in_terminal()

    def log_init_images(self):
        """
        logging the augmentation results and split data images that are made before training
        """
        pass

    # -------- helper functions --------

    def on_early_stop(self):
        wandb.log({"early_stop": True})
        wandb.finish()
        self.show_in_terminal()

    def on_end(self):
        wandb.log({"early_stop": False})
        wandb.finish()
        self.show_in_terminal()

    def handle_watch_model(self, model, criterion):
        wandb.watch(model, criterion=criterion, log="all")

    def get_run_name(self):
        return str(wandb.run.name)

    def get_context(self, config, project):
        wandb.login(key="629ea7adb0e1489b5e303ea35e3175a9bea918e4")
        context = wandb.init(config=config, project=project, settings=wandb.Settings(start_method="fork"))
        return context

    # ---------------- show ---------------
    def show_in_terminal(self):
        print("|" + " | ".join([f"{k}: {v.__round__(3)}" for k, v in self.items.items()]) + "|")

    # ---------------- moving average ---------------
    def moving_average_record(self, name, value, window_size=10):
        if name not in self.moving_average:
            self.moving_average[name] = [value]
        else:
            self.moving_average[name].append(value)
            if len(self.moving_average[name]) > window_size:
                self.moving_average[name].pop(0)

    def moving_average_get(self, name):
        return sum(self.moving_average[name]) / len(self.moving_average[name])

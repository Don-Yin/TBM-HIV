"""
very proud of this shit: grid search handler
"""

from pathlib import Path
from itertools import product
from typing import Any
from rich import print
from monai.networks.layers.factories import Act
from src.train_main import main
import json

import argparse

parser = argparse.ArgumentParser(description="Grid search handler")
parser.add_argument("--run_with", type=int, default=-1, required=False, help="Run with the parameters of the index if exists")
parser.add_argument("--mark_submitted", type=int, default=-1, required=False, help="mark a run as submitted to jade")
parser.add_argument("--check_node_type", type=int, default=-1, help="checking the node for jade of idx")
parser.add_argument("--len", action="store_true", help="check the total number of combinations")

args = parser.parse_args()


class GridSearchHandler:
    def __init__(self, configurations: dict):
        """
        e.g.,
        config = {
            "learning_rate": [1e-4, 1e-5],
            "batch_size": [4, 8],
            "blocks": [[6, 12, 24, 32], [6, 12, 24, 48]],
            ...
        }
        """
        self.configurations = configurations
        self._make_combinations()

    # -------- main functions --------
    def run_with_idx(self, idx):
        main(self.combinations[idx], epochs=1000, epochs_min=70, logger_name="tensorboard")

    def get_node_type(self, idx):
        """
        This is for JADE only; returns the node type: small / big
            custom conditions:
        """
        if self.combinations[idx]["densenet_block_config"] == (6, 12, 64, 48):
            return "BIG"
        return "SMALL"

    # -------- jade registry --------
    def init_registry(self):
        registry = {str(i): {"submitted": False, "node": self.get_node_type(i)} for i in range(len(self))}
        save_at = Path("runs")
        save_at.mkdir(parents=True, exist_ok=True)
        with open(save_at / "registry.json", "w") as f:
            json.dump(registry, f, indent=4)

    def mark_idx_as(self, idx, submitted: bool):
        registry_path = Path("runs/registry.json")
        if not registry_path.exists():
            self.init_registry()

        with open(registry_path, "r") as f:
            registry = json.load(f)

        registry[str(idx)]["submitted"] = submitted

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=4)

    # -------- helper functions --------
    def _make_combinations(self):
        keys, values = zip(*self.configurations.items())
        self.combinations = [dict(zip(keys, v)) for v in product(*values)]

    # -------- dunder methods --------
    def __call__(self, actions: dict):
        if actions["len"]:
            print(len(self))

        if actions["check_node_type"] != -1:
            print(self.get_node_type(actions["check_node_type"]))

        if actions["mark_submitted"] != -1:
            self.mark_idx_as(actions["mark_submitted"], True)

        if actions["run_with"] != -1:
            self.run_with_idx(actions["run_with"])

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        print(f"Getting item index {idx} from total {len(self.combinations)} combinations")
        return self.combinations[idx]


if __name__ == "__main__":
    configurations = {
        # -------- mlp: fix this for now --------
        "mlp_input_size": [56],  # 56 with full varibles and 53 for without gcs
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
        # (6, 12, 24, 16), (6, 12, 32, 32), (6, 12, 48, 32) / (6, 12, 64, 48), 121, 169, 201, 264; it looks like the size doesn't matter when we use the tbm grade as the label; but i'd still like to try different sizes when lesion type is the label
        "densenet_block_config": [(6, 12, 64, 48)],
        "densenet_bn_size": [4],  # bottleneck; channel limit = bn_size * growth_rate
        "densenet_act": [Act.LEAKYRELU],  # "relu"
        "densenet_norm": ["batch"],
        # looks like dropout doesn't matter too much; 0.05, 0.15
        "densenet_dropout_prob": [0.15],
        # -------- overall --------
        "meta_batch_size": [4],
        "meta_learning_rate": [3e-4],  # current best larger then 3e-4
        "meta_input_mode": ["BOTH", "ONLY_IMAGE", "ONLY_CLINICAL"],
        "meta_use_weighted_loss": [True],  # better as false, strange
        "meta_augment": [True],  # no effect on the results; the settings are amended, try again
        "meta_combined_model_linear_layer_size": [[64, 32]],
        "meta_use_label": ["LESION_TYPE", "TBM"],
    }

    grid_search_handler = GridSearchHandler(configurations)
    grid_search_handler(vars(args))

import os
import torch
from operator import itemgetter
from torch import nn
from torch.nn.init import kaiming_normal_
from rich import print
from natsort import natsorted
from src.utils.general import banner


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()


def apply_kaiming_initialization(model):
    banner("Initializing model with Kaiming Normal.")
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            kaiming_normal_(module.weight, nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def load_checkpoint(checkpoint_path, model):
    checkpoint_files = os.listdir(checkpoint_path)
    checkpoint_files = [(f, os.path.getmtime(checkpoint_path / f)) for f in checkpoint_files]
    checkpoint_files.sort(key=itemgetter(1), reverse=True)
    latest_checkpoint = checkpoint_files[0][0]
    model.load_state_dict(torch.load(checkpoint_path / latest_checkpoint))
    return model


def save_checkpoint(model, checkpoint_path):
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    existing = os.listdir(checkpoint_path)
    latest_existing = natsorted(existing)[-1] if len(existing) > 0 else None
    if latest_existing is not None:
        save_name = latest_existing.split("_")[1].split(".")[0]
        save_name = int(save_name) + 1
        try:
            os.remove(checkpoint_path / latest_existing)
        except:
            pass
    else:
        save_name = 0
    torch.save(model.state_dict(), checkpoint_path / f"checkpoint_{save_name}.pth")

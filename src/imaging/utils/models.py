import torch
import torch.nn as nn
from src.imaging.utils.general import device
from monai.networks.nets import DenseNet
from monai.networks.layers.factories import Act


class MLP(nn.Module):
    def __init__(
        self,
        input_size=48,
        output_size=4,
        hidden_sizes=[32, 16, 8],
        dropout_prob=0.1,
        activation_function=Act.LEAKYRELU,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        # Define the layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
        self.layers.append(Act[activation_function]())
        self.layers.append(nn.Dropout(p=dropout_prob))

        # Hidden layers
        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            self.layers.append(Act[activation_function]())
            self.layers.append(nn.Dropout(p=dropout_prob))

        # Output layer
        self.layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        config_mlp = {i.replace("mlp_", ""): config[i] for i in config if i.startswith("mlp_")}
        config_densenet = {i.replace("densenet_", ""): config[i] for i in config if i.startswith("densenet_")}

        if config["meta_use_label"] == "TBM":
            output_classes = 3  # tbm = 3, lesion type = 6
        elif config["meta_use_label"] == "LESION_TYPE":
            output_classes = 6

        self.hidden_layers_size = config["meta_combined_model_linear_layer_size"]
        self.model_mlp = MLP(**config_mlp).to(device)
        self.model_densenet = DenseNet(**config_densenet).to(device)

        self.input_mode = config["meta_input_mode"]
        if self.input_mode == "BOTH":
            first_layer_input_size = config_densenet["out_channels"] + config_mlp["output_size"]
        elif self.input_mode == "ONLY_IMAGE":
            first_layer_input_size = config_densenet["out_channels"]
        elif self.input_mode == "ONLY_CLINICAL":
            first_layer_input_size = config_mlp["output_size"]

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(first_layer_input_size, self.hidden_layers_size[0]))
        self.layers.append(Act[config_mlp["activation_function"]]())
        self.layers.append(nn.Dropout(p=config_mlp["dropout_prob"]))

        for i in range(len(self.hidden_layers_size) - 1):
            self.layers.append(nn.Linear(self.hidden_layers_size[i], self.hidden_layers_size[i + 1]))
            self.layers.append(Act[config_mlp["activation_function"]]())
            self.layers.append(nn.Dropout(p=config_mlp["dropout_prob"]))

        self.layers.append(nn.Linear(self.hidden_layers_size[-1], output_classes))

    def forward(self, image, clinical_data):
        if self.input_mode == "BOTH":
            output_image, output_mlp = self.model_densenet(image), self.model_mlp(clinical_data)
            combined = torch.cat((output_image, output_mlp), dim=1)
            for layer in self.layers:
                combined = layer(combined)
            output = combined

        elif self.input_mode == "ONLY_IMAGE":
            output_image = self.model_densenet(image)
            for layer in self.layers:
                output_image = layer(output_image)
            output = output_image

        elif self.input_mode == "ONLY_CLINICAL":
            output_mlp = self.model_mlp(clinical_data)
            for layer in self.layers:
                output_mlp = layer(output_mlp)
            output = output_mlp

        return output


if __name__ == "__main__":
    import torchsummary

    # Instantiate the model
    model = MLP()
    torchsummary.summary(model, (48,))
    dummy_input = torch.randn(1, 48)
    print(model(dummy_input))

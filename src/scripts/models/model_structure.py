from src.diagram.sankey import SankeyDiagram, SankeyData, SankeyModel, SankeyChain, SankeyCluster, SankeyLayer
from pathlib import Path

if __name__ == "__main__":
    # --------
    diagram = SankeyDiagram(orientation="v", width=950, height=950)

    # --------
    node_input_image = SankeyData("Image (193, 229, 193)").hook(diagram).set_width(64)
    node_input_clinical_data = SankeyData("Clinical Data (56)").hook(diagram).set_width(56)

    # node_densenet = SankeyModel("DenseNet").hook(diagram)
    node_densenet = SankeyChain(
        [
            SankeyModel("DenseBlock 6").hook(diagram).set_nature("denseblock").set_width(56),
            SankeyModel("Transition").hook(diagram).set_nature("transition").set_width(56),
            SankeyModel("DenseBlock 12").hook(diagram).set_nature("denseblock").set_width(56),
            SankeyModel("Transition").hook(diagram).set_nature("transition").set_width(56),
            SankeyModel("DenseBlock 24").hook(diagram).set_nature("denseblock").set_width(56),
            SankeyModel("Transition").hook(diagram).set_nature("transition").set_width(56),
            SankeyModel("DenseBlock 16").hook(diagram).set_nature("denseblock").set_width(56),
            SankeyLayer("Latent Space").hook(diagram).set_nature("latent").set_width(56),
        ]
    )

    node_mlp = SankeyChain(
        [
            SankeyLayer(f"Linear 32").hook(diagram).set_nature("linear").set_width(32),
            SankeyLayer(f"Leaky ReLU").hook(diagram).set_nature("relu").set_width(32),
            SankeyLayer(f"Dropout").hook(diagram).set_nature("dropout").set_width(32),
            SankeyLayer(f"Linear 24").hook(diagram).set_nature("linear").set_width(24),
            SankeyLayer(f"Leaky ReLU").hook(diagram).set_nature("relu").set_width(24),
            SankeyLayer(f"Dropout").hook(diagram).set_nature("dropout").set_width(24),
            SankeyLayer(f"Linear 16").hook(diagram).set_nature("linear").set_width(16),
            SankeyLayer("Latent Space").hook(diagram).set_nature("latent").set_width(8),
        ]
    )

    code_concatenate = SankeyLayer("Concatenate").hook(diagram).set_nature("concatenate").set_width(64)

    node_combined_linear_model = SankeyChain(
        [
            SankeyLayer(f"Linear 64").hook(diagram).set_nature("linear").set_width(64),
            SankeyLayer(f"Leaky ReLU").hook(diagram).set_nature("relu").set_width(64),
            SankeyLayer(f"Dropout").hook(diagram).set_nature("dropout").set_width(64),
            SankeyLayer(f"Linear 32").hook(diagram).set_nature("linear").set_width(32),
        ]
    )

    node_output = SankeyLayer("Output").hook(diagram).set_nature("output").set_width(6)

    # --------
    node_input_image.connect_to(node_densenet)
    node_input_clinical_data.connect_to(node_mlp)
    node_densenet.connect_to(code_concatenate)
    node_mlp.connect_to(code_concatenate)

    code_concatenate.connect_to(node_combined_linear_model)
    node_combined_linear_model.connect_to(node_output)

    # --------
    save_to = Path("results", "images", "model", "sankey_diagram.png")
    save_to.parent.mkdir(parents=True, exist_ok=True)
    diagram.draw()
    diagram.save_as_png(save_to)

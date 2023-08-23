from typing import Any
import plotly.graph_objects as go
import seaborn as sns


class Color:
    def __init__(self, idx: int):
        self.idx = idx
        self.COLORS = sns.color_palette("pastel")

    def get(self, alpha=1.0) -> Any:
        return self.convert_to_rgba_format(self.COLORS[self.idx], alpha)

    def convert_to_rgba_format(self, color, alpha=0.3):
        """Convert RGB values from 0-1 float to 0-255 integer."""
        return "rgba({},{},{},{})".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), alpha)


def calculate_flow(nodes, links):
    """Adjust the link values to make them cover 100% of node width"""
    node_flow = {node.idx: 0 for node in nodes}

    for node in nodes:
        has_incoming = any(link["target"] == node.idx for link in links)
        if not has_incoming:
            node_flow[node.idx] = node.custom_width

    for link in links:
        source_flow = node_flow[link["source"]]
        num_children = len([l for l in links if l["source"] == link["source"]])
        link_value = source_flow / num_children
        link["value"] = link_value
        node_flow[link["target"]] += link_value

    return links


# def calculate_flow(nodes, links):
#     """Adjust the link values to make them cover 100% of node width"""
#     node_flow = {node.idx: 0 for node in nodes}  # Initialize all nodes with flow of 0

#     # Distribute the custom_width of the source node amongst its children (outgoing links)
#     for link in links:
#         num_children = len([l for l in links if l["source"] == link["source"]])
#         link_value = nodes[link["source"]].custom_width / num_children
#         link["value"] = link_value
#         node_flow[link["target"]] += link_value  # Accumulate width on target nodes

#     return links


class SankeyDiagram:
    def __init__(self, orientation="h", width=950, height=1200):
        self.current_idx = 0
        self.orientation = orientation
        self.nodes = []
        self.natures = []

        self.width = width
        self.height = height

    # -------- methods --------s
    def add_node(self, node):
        node.idx = self.current_idx
        self.current_idx += 1
        self.nodes.append(node)

    # -------- save related --------
    def save_as_png(self, file_path: str):
        """Saves the diagram as a PNG to the provided file path."""
        self.fig.write_image(file_path, format="png")

    def show(self):
        self.fig.show()

    def draw(self):
        links = []
        for node in self.nodes:
            for child in node.children:
                if isinstance(child, SankeyCluster) or isinstance(child, SankeyChain):
                    for child_node in child.nodes:
                        links.append(
                            {"source": node.idx, "target": child_node.idx, "value": 1, "color": node.color.get(alpha=0.2)}
                        )
                else:
                    links.append({"source": node.idx, "target": child.idx, "value": 1, "color": node.color.get(alpha=0.2)})

        links = calculate_flow(self.nodes, links)

        self.fig = go.Figure(
            go.Sankey(
                node=dict(
                    pad=270,
                    thickness=36,
                    line=dict(color="black", width=2),
                    label=[node.name for node in self.nodes],
                    color=[node.color.get(alpha=0.5) for node in self.nodes],
                    # x=[node.x for node in self.nodes],
                    # y=[node.y for node in self.nodes],
                ),
                link=dict(
                    # arrowlen=81,
                    source=[link["source"] for link in links],
                    target=[link["target"] for link in links],
                    value=[link["value"] for link in links],
                    # color=[link["color"] for link in links],
                    color="rgba(0,0,0,0.2)",
                ),
                orientation=self.orientation,
                arrangement="snap",
            )
        )

        self.fig.update_layout(
            # title="Model Structure",
            font=dict(family="Arial", size=24),
            width=self.width,
            height=self.height,
            margin=dict(l=5, r=5, b=5, t=5),  # Setting a top margin to make space for the title
        )


# --------
class SankeyChain:
    def __init__(self, nodes: list):
        self.nodes = nodes
        for i in range(len(self.nodes) - 1):
            self.nodes[i].connect_to(self.nodes[i + 1])
        self.idx = [node.idx for node in self.nodes]

    def connect_to(self, other):
        self.nodes[-1].connect_to(other)
        return self


class SankeyCluster:
    def __init__(self, nodes: list):
        self.nodes = nodes
        self.idx = [node.idx for node in self.nodes]


# --------
class SankeyNode:
    def __init__(self, name: str):
        self.name = name
        self.children = []
        self.custom_width = 10
        self.nature = "node"

    def connect_to(self, node):
        if isinstance(node, SankeyCluster):
            self.children += node.nodes
        elif isinstance(node, SankeyChain):
            self.children.append(node.nodes[0])
        else:
            self.children.append(node)

        self.remove_duplicate()
        return self

    def remove_duplicate(self):
        seen = set()
        self.children = [node for node in self.children if node.idx not in seen and not seen.add(node.idx)]

    def hook(self, diagram: SankeyDiagram):
        self.diagram = diagram
        self.diagram.add_node(self)
        self.set_nature(self.nature)
        return self

    def set_width(self, width: float):
        self.custom_width = width
        return self

    def set_nature(self, nature: str):
        self.nature = nature
        if self.nature not in self.diagram.natures:
            self.diagram.natures.append(self.nature)

        # -------- amend property by nature --------
        self.color = Color(self.diagram.natures.index(self.nature))
        return self


class SankeyModel(SankeyNode):
    def __init__(self, name: str):
        super().__init__(name)


class SankeyLayer(SankeyNode):
    def __init__(self, name: str):
        super().__init__(name)


class SankeyData(SankeyNode):
    def __init__(self, name: str):
        super().__init__(name)

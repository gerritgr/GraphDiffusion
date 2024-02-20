import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch import nn



############################################
# GD object
############################################


from graphdiffusion.distance import GraphDistance, GraphDistanceWasserstein, GraphDistanceAssignment





import torch
from torch_geometric.datasets import QM9

# Load the QM9 dataset
dataset = QM9(root="./data/QM9")

# Print some dataset statistics
print(f"Dataset: {dataset}:")
print("====================")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")


# Get the first graph object from the dataset
data = dataset[0]  # Get the first graph object.

print(data)
print("=============================================================")

# Graph attributes
print(f"Number of nodes (atoms): {data.num_nodes}")
print(f"Number of edges (bonds): {data.num_edges}")
print(f"Number of node features (atom types): {data.num_node_features}")
print(f"Edge index (connectivity): \n{data.edge_index}")


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx

# Convert a PyG graph object to a NetworkX graph
graph = to_networkx(data, to_undirected=True)

# Draw the graph
#plt.figure(figsize=(10, 7))
#nx.draw(graph, with_labels=True, node_color="lightblue", edge_color="gray")
#plt.title("Graph Representation of a Molecule")
#plt.savefig("images/example7_molecule_graph.png")

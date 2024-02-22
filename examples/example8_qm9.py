import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch import nn


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx


from graphdiffusion.distance import GraphDistance, GraphDistanceWasserstein, GraphDistanceAssignment

import torch
from torch_geometric.datasets import QM9

from graphdiffusion.utils import *

############################################
# GD object
############################################



# Load the QM9 dataset
dataset = QM9(root="./data/QM9")

# Print some dataset statistics
print(f"Dataset: {dataset}:")
print("====================")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")


# Get the first graph object from the dataset
data = dataset[1021]  # Get the first graph object.

print(data)
print("=============================================================")

# Graph attributes
print(f"Number of nodes (atoms): {data.num_nodes}")
print(f"Number of edges (bonds): {data.num_edges}")
print(f"Number of node features (atom types): {data.num_node_features}")
print(f"Edge index (connectivity): \n{data.edge_index}")



print(data)
print(data.x)
print(data.edge_index)
print(data.edge_attr)


# hydrogen, carbon, Nitrogen, Oxygen, Fluorine



def transform_qm9(data):
    data_new = data.clone()
    data_new.x = data_new.x[:, :6]
    return data_new

data = transform_qm9(data)
print("data new")
print(data)
print(data.x)
print(data.edge_index)
print(data.edge_attr)






import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# Assume 'data' is your PyG graph data object

# Define a list of colors corresponding to the atom types represented in the one-hot encoding
# Adjust these colors as needed to match the atom types you're interested in
# For example: [Hydrogen, Carbon, Nitrogen, Oxygen, Fluorine]
colors = ['white', 'black', 'blue', 'red', 'green']  # Example colors for 5 atom types

# Function to map one-hot encoded atom types to colors
def one_hot_to_color(one_hot_vector, colors):
    # Find the index of the '1' in the one-hot vector, which indicates the atom type
    atom_type_index = one_hot_vector.argmax()
    # Return the corresponding color
    return colors[atom_type_index]

# Convert one-hot encodings to colors for each node
node_colors = [one_hot_to_color(data.x[i, :5], colors) for i in range(data.num_nodes)]

# Convert the PyG graph object to a NetworkX graph
graph = to_networkx(data, to_undirected=True)

# Custom drawing
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(graph)  # Position nodes using the spring layout

# Draw nodes with a black border
nx.draw_networkx_nodes(graph, pos, node_color=node_colors, edgecolors='black', linewidths=2, node_size=700)

# Draw edges
nx.draw_networkx_edges(graph, pos, edge_color="gray")

# Draw labels with white text
labels = {i: i for i in range(len(graph))}
nx.draw_networkx_labels(graph, pos, labels, font_color='gray')

plt.title("Graph Representation of a Molecule with Atom-specific Colors")
plt.axis('off')  # Turn off the axis
#plt.savefig(create_path("images/example8/molecule_graph.png"))







from graphdiffusion.plotting import * 

fig, ax = plt.subplots(figsize=(10, 7))
plot_pyg_graph(data, ax, remove_hydrogens=True)
plt.savefig(create_path("images/example8/molecule_graph.png"))



fig, ax = plt.subplots(figsize=(10, 7))
plot_pyg_graph(data, ax, remove_hydrogens=False)
plt.savefig(create_path("images/example8/molecule_graph_with_h.png"))








from graphdiffusion.pipeline import *
from graphdiffusion.bridge import *
from graphdiffusion.degradation import *
from graphdiffusion.reconstruction import *
from graphdiffusion.distance import *
from graphdiffusion.utils import *

set_all_seeds(1234)


from torch_geometric.loader import DataLoader

# Load the QM9 dataset

batch_size = 2  # You can adjust the batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def degradation_obj(data, t, pipeline):
    return data

pipeline = PipelineVector(node_feature_dim=2, level="DEBUG",     degradation_obj=degradation_obj,)


pipeline.visualize_foward(
    data=dataloader,
    outfile="images/example8/dummy_forward.pdf",
    num=25,
    plot_data_func=plot_pyg_graph,
)

data = dataset[1021] #dataset[101]
data = remove_hydrogens_from_pyg(data)
data.x = data.x[:,0:5]
print("\n"*6)
print("before transform ", data)

fig, ax = plt.subplots(figsize=(10, 7))
plot_pyg_graph(data, ax, remove_hydrogens=False)
plt.savefig(create_path("images/example8/molecule_edgegraph.png"))

data = inflate_graph(data)
print("after transform ", data)
print("print(data.x)")
print(data.x)

fig, ax = plt.subplots(figsize=(10, 7))
plot_pyg_graph(data, ax, remove_hydrogens=False)
plt.savefig(create_path("images/example8/molecule_edgegraph2.png"))

print(data)
print("print(data.x)")
print(str(data.x.numpy()))
print(data.edge_index.t())


print(data.node_mask, data.x[data.node_mask])
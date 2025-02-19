import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch import nn

# Step 1: Define the Toy Graph
# Define edges - for a simple graph, let's say we have edges between nodes 0-1, 1-2, and 2-0 (creating a triangle)
edge_index = torch.tensor([[0, 1, 2, 1], [1, 2, 0, 0]], dtype=torch.long)  # Directed edges

# Node features - for simplicity, let's use one-hot encoded features
x = torch.eye(3)  # 3 nodes with 3 features each

# Create a graph data object
data1 = Data(x=x, edge_index=edge_index)
data2 = Data(x=x * 1.1, edge_index=edge_index)
data3 = Data(x=x * 5.1, edge_index=edge_index)



############################################
# GD object
############################################


from graphdiffusion.distance import GraphDistance, GraphDistanceWasserstein, GraphDistanceAssignment

print("\n"*10)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], 
                           [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
num_nodes = 4  # Total number of nodes in the graph
node_features = torch.randn(num_nodes, 1)  # Example feature vectors of dimension 5
graph1 = Data(x=node_features, edge_index=edge_index)
node_permutation = torch.randperm(num_nodes)
permuted_node_features = node_features[node_permutation]
permuted_edge_index = node_permutation[edge_index]
graph2 = Data(x=permuted_node_features, edge_index=permuted_edge_index)


print("again naive distance: ", torch.norm(graph1.x - graph2.x, p=2))
print("comapre ")
print(graph1.x)
print("with")
print(graph2.x)
graph2.x.requires_grad = True

print("graph distance naive: ",  torch.norm(graph1.x - graph2.x, p=2))

dist = GraphDistance()
print("graph distance: ", dist(graph1, graph2))

dist = GraphDistanceWasserstein()
print("graph distance wasserstein: ", dist(graph1, graph2))

dist = GraphDistanceAssignment()
print("graph distance assign: ", dist(graph1, graph2))


############################################
# qm9
############################################


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

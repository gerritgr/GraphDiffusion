
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch import nn
from graphdiffusion.utils import *

from torch_geometric.nn import GCN

def permute_graph(data, node_permutation):
    if isinstance(data, torch.Tensor):
        x = data
        return x[node_permutation]

    permuted_node_features = data.x[node_permutation]
    permuted_edge_index = node_permutation[data.edge_index]
    data = data.clone()
    data.x = permuted_node_features
    data.edge_index = permuted_edge_index
    return data


# Gen graphs

edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], 
                           [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
num_nodes = 4  # Total number of nodes in the graph
node_features = torch.tensor([0.1,1.1,2.1,3.1]).view(-1,1) + torch.randn(num_nodes, 1)*0.01  # Example feature vectors of dimension 5
graph1 = Data(x=node_features, edge_index=edge_index)

node_permutation = torch.tensor([0, 2, 3, 1])
node_permutation_reverse = torch.tensor([0, 3, 1, 2])

graph2 = permute_graph(graph1, node_permutation)

graph3 = permute_graph(graph2, node_permutation_reverse)

print("\n"*10)
print("input graphs", node_permutation)
print(graph1.x)
print(graph2.x)
print(graph3.x)
print("input graphs egde", node_permutation)
print(graph1.edge_index)
print(graph2.edge_index)
print(graph3.edge_index)

print("-----------")
model = GCN(1, 4, 4, 1)  # use uneven number like 1,3,3,1 and everything breaks
model.eval()
with torch.no_grad():
    x = model(graph1.x * 10, graph1.edge_index)
    print('graph 1 out \n' ,x)
    x2 = model(graph2.x * 10, graph2.edge_index)
    print('graph 2 out \n', x2)
    x3 = permute_graph(x2, node_permutation_reverse)
    print('graph 2 back \n', x3)


dist = torch.norm(x - x3, p=2)
print("permuteated back distance: ", dist)

assert dist.item()<1e-5


print("========================")
############################################
# GD object
############################################
from graphdiffusion.distance import GraphDistance, GraphDistanceWasserstein, GraphDistanceAssignment
graph1.x.requires_grad = True

dist = GraphDistance()
d_g = dist(graph1, graph2)

dist = GraphDistanceWasserstein()
d_w = dist(graph1, graph2)

dist = GraphDistanceAssignment()
d_da = dist(graph1, graph2)

print("graph distance: ", d_g)
print("graph distance wasserstein: ", d_w)
print("graph distance assign: ", d_da)

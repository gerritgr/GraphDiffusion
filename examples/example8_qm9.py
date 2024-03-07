import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch import nn


import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
from graphdiffusion.plotting import * 

from graphdiffusion.distance import GraphDistance, GraphDistanceWasserstein, GraphDistanceAssignment

import torch
from torch_geometric.datasets import QM9

from graphdiffusion.utils import *


from graphdiffusion.pipeline import *
from graphdiffusion.bridge import *
from graphdiffusion.degradation import *
from graphdiffusion.reconstruction import *
from graphdiffusion.distance import *
from graphdiffusion.utils import *


from torch_geometric.loader import DataLoader


set_all_seeds(1234)


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





# Load the QM9 dataset
def pre_process_qm9(data):
    data.x = data.x[:,0:5]
    data = remove_hydrogens_from_pyg(data)
    data = inflate_graph(data)
    return data

def gen_dataloader(batch_size = 1):
    import gzip, pickle
    #try:
    #    print("try to load data")
    #    with gzip.open("./data/QM9/transformed_dataset.pkl.gz", 'rb') as f:
    #        transformed_dataset = pickle.load(f)
    #    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)
    #    return dataloader
    #except:
    #    print("generate data")
    transformed_dataset = [pre_process_qm9(data) for data in tqdm(dataset[:1000])]
    #transformed_dataset_path = "./data/QM9/transformed_dataset.pkl.gz"
    #with gzip.open(transformed_dataset_path, 'wb') as f:
    #    pickle.dump(transformed_dataset, f)
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

dataloader = gen_dataloader()



def degradation_obj_with_reduce(data, t, pipeline): #TODO batches
    data = data.clone()
    new_x = data.x + t * 3 * torch.randn_like(data.x)
    if 'batch' in data:
        original_node_feature_dim = data.original_node_feature_dim[0]
        original_edge_feature_dim = data.original_edge_feature_dim[0]
    else:
        original_node_feature_dim = data.original_node_feature_dim
        original_edge_feature_dim = data.original_edge_feature_dim

    data.x[data.node_mask,1:original_node_feature_dim+1] = new_x[data.node_mask,1:original_node_feature_dim+1]
    data.x[data.edge_mask,1:original_edge_feature_dim+1] = new_x[data.edge_mask,1:original_edge_feature_dim+1]
    data= reduce_graph(data)
    return data


pipeline = PipelineVector(node_feature_dim=2, level="DEBUG",     degradation_obj=batchify_pyg_transform(degradation_obj_with_reduce)) #remove_hydrogens_from_pyg


pipeline.visualize_foward(
    data=dataloader,
    outfile="images/example8/dummy_forward.pdf",
    num=25,
    plot_data_func=plot_pyg_graph,
)



data = dataset[101] #dataset[101] #1021

data = pre_process_qm9(data)

#data = inflate_graph(data)
print("after transform ", data)
print("print(data.x)")
print(data.x)

fig, ax = plt.subplots(figsize=(10, 7))
plot_pyg_graph(data, ax)
plt.savefig(create_path("images/example8/molecule_inflated_graph.png"))

print(data)
print("print(data.x)")
print(str(data.x.numpy()))
print(data.edge_index.t())


print(data.node_mask, data.x[data.node_mask])


data = reduce_graph(data)


print(data)

fig, ax = plt.subplots(figsize=(10, 7))
plot_pyg_graph(data, ax)
plt.savefig(create_path("images/example8/molecule_inflated_and_reduced_graph.png"))






################
#### learning
################

dataloader = gen_dataloader(batch_size = 2)




def degradation_obj(data, t, pipeline): #TODO batches
    data = data.clone()
    new_x = data.x + t * 3 * torch.randn_like(data.x)
    #if 'batch' in data:
    #    original_node_feature_dim = data.original_node_feature_dim[0]
    #    original_edge_feature_dim = data.original_edge_feature_dim[0]
    #else:
    original_node_feature_dim = data.original_node_feature_dim
    original_edge_feature_dim = data.original_edge_feature_dim
    data.x[data.node_mask,1:original_node_feature_dim+1] = new_x[data.node_mask,1:original_node_feature_dim+1]
    data.x[data.edge_mask,1:original_edge_feature_dim+1] = new_x[data.edge_mask,1:original_edge_feature_dim+1]
    return data

pipeline = PipelineVector(node_feature_dim=2, level="DEBUG", degradation_obj=batchify_pyg_transform(degradation_obj)) #remove_hydrogens_from_pyg


pipeline.visualize_foward(
    data=dataloader,
    outfile="images/example8/dummy_forward2.pdf",
    num=25,
    plot_data_func=plot_pyg_graph,
)
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
    transformed_dataset = [data for data in transformed_dataset if data.edge_index.numel() > 1]
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
#### plotting
################

print("\n"*10)

dataloader = gen_dataloader(batch_size = 4)


for data_batch in dataloader:
    print(data_batch)
    print(data_batch.x)
    print(data_batch.edge_index)
    print(data_batch.edge_attr)
    break




def degradation_obj(data, t, pipeline): #TODO batches
    data = data.clone()
    if isinstance(t, torch.Tensor):
        t = t[0]
    t = torch.ones((data.x.shape[0], 1), device=pipeline.device) * t #todo fix batches
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



################
#### learning
################

from torch_geometric.nn import GCN, GIN, BatchNorm, PNA
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
class graph_reconstructionGIN(nn.Module):
    def __init__(self, node_feature_dim, time_dim):
        super(graph_reconstructionGIN, self).__init__()
        hidden_channels = 32
        self.gnn =  GIN(in_channels= node_feature_dim + time_dim, hidden_channels = hidden_channels, num_layers = 4, out_channels=node_feature_dim, dropout=0.1, train_eps=True, jk='cat', norm=BatchNorm(hidden_channels))


    def forward(self, data, t, condition=None, pipeline=None, *args, **kwargs):
        data = data.clone()
        if isinstance(t, torch.Tensor):
            t = t[0]
        t = torch.ones((data.x.shape[0], 1), device=data.x.device) * t #todo fix batches
        x_in = torch.cat((data.x , t), dim=1)
        new_x = self.gnn(x_in, data.edge_index)
        original_node_feature_dim = data.original_node_feature_dim[0]
        original_edge_feature_dim = data.original_edge_feature_dim[0]
        data.x[data.node_mask,1:original_node_feature_dim+1] = new_x[data.node_mask,1:original_node_feature_dim+1]
        data.x[data.edge_mask,1:original_edge_feature_dim+1] = new_x[data.edge_mask,1:original_edge_feature_dim+1]
        return data
    
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx, degree
def dataset_to_degree_bin(train_dataset):
  """
  Convert a dataset to a histogram of node degrees (in-degrees).
  Load from file if available; otherwise, compute from the dataset.
  """

  # Assert that the dataset is provided.
  assert(train_dataset is not None)

  # Compute the maximum in-degree in the training data.
  max_degree = -1
  for data in train_dataset:
    #data = data.to(DEVICE)
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    max_degree = max(max_degree, int(d.max()))

  # Create an empty histogram for degrees.
  deg = torch.zeros(max_degree + 1, dtype=torch.long)

  # Populate the histogram with data from the dataset.
  for data in train_dataset:
    #data = data.to(DEVICE)
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())

  # Save the computed histogram to a file.
  #write_file("deg.pickle", deg.cpu())

  return deg
class graph_reconstruction(nn.Module):
    def __init__(self, node_feature_dim, time_dim, train_dataset, towers=2, normalization=True, depth=6, dropout=0.05, pre_post_layers=1, hidden_channels=32):
        super(graph_reconstruction, self).__init__()
        self.sigmoid = nn.Sigmoid()

        # Adjust hidden channels for the given towers.
        hidden_channels = towers * ((hidden_channels // towers) + 1) # must match

        # Calculate input and output channels.
        in_channels = node_feature_dim + time_dim
        out_channels = node_feature_dim

        # Get degree histogram for the dataset
        deg = dataset_to_degree_bin(train_dataset)

        # Set aggregators and scalers for the PNA layer.
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        # Create a normalization layer if required.
        self.normalization = BatchNorm(hidden_channels) if normalization else None

        # Define the PNA layer.
        self.pnanet = PNA(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=depth,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            dropout=dropout,
            towers=towers,
            norm=self.normalization,
            pre_layers=pre_post_layers,
            post_layers=pre_post_layers
        )

        # Define the final MLP layer.
        self.final_mlp = Seq(
            Lin(hidden_channels, hidden_channels),
            nn.ReLU(),
            Lin(hidden_channels, hidden_channels),
            nn.ReLU(),
            Lin(hidden_channels, hidden_channels),
            nn.ReLU(),
            Lin(hidden_channels, out_channels)
        )

    def forward(self, data, t, condition=None, pipeline=None, *args, **kwargs):
        """
        Perform a forward pass through the PNAnet.
        """
        data = data.clone()
        x_in = data.x
        edge_index = data.edge_index
        TIME_FEATURE_DIM = 1
        row_num = x_in.shape[0]
        t = t.view(-1, TIME_FEATURE_DIM)
        x = torch.concat((x_in, t), dim=1)

        x = self.pnanet(x, edge_index)
        x = self.final_mlp(x)

        # Assertions for sanity checks
        assert(x.numel() > 1)
        assert(x.shape[0] == row_num)

        original_node_feature_dim = data.original_node_feature_dim[0]
        original_edge_feature_dim = data.original_edge_feature_dim[0]
        data.x[data.node_mask,1:original_node_feature_dim+1] = x[data.node_mask,1:original_node_feature_dim+1]
        data.x[data.edge_mask,1:original_edge_feature_dim+1] = x[data.edge_mask,1:original_edge_feature_dim+1]
        return data

train_dataset = [data for data in gen_dataloader(batch_size = 1)]
model = graph_reconstruction(5,1, train_dataset)

def train_graph_epoch(dataloader, pipeline, optimizer):
    model = pipeline.get_model()
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = pipeline.preprocess(batch) 

        if isinstance(batch, list) or isinstance(batch, tuple):
            batch, _ = batch # ignore label

        batch = batch.to(pipeline.device)
        # Generate random tensor 't' for degradation
        t = torch.ones(batch.x.shape[0], device=pipeline.device) * torch.rand(1)

        optimizer.zero_grad()
        # Apply degradation and reconstruction from the pipeline
        batch_with_noise = pipeline.degradation(batch, t)
        batch_reconstructed = pipeline.reconstruction(batch_with_noise, t)

        # Compute loss using the pipeline's distance method
        loss = pipeline.distance(batch, batch_reconstructed)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate average loss for the epoch
    average_loss = total_loss / len(dataloader)
    return average_loss

train_obj = VectorTrain(train_epoch_func=train_graph_epoch)



class VectorBridgeGraph(nn.Module):
    def __init__(self):
        super(VectorBridgeGraph, self).__init__()

    def forward(self, data_now, data_prediction, t_now, t_query, pipeline, vectorbridge_magnitude_scale=None, vectorbridge_rand_scale=None):
        vectorbridge_magnitude_scale = (
            pipeline.config.vectorbridge_magnitude_scale or 1.0
        )  # vectorbridge_magnitude_scale should be taken automatically from the config, this does not work currently somehow
        vectorbridge_rand_scale = pipeline.config.vectorbridge_rand_scale or 1.0
        vectorbridge_magnitude_scale = 1.0
        vectorbridge_rand_scale = 2.0
        data_new = data_now.clone()
        data_now = data_now.x
        data_prediction = data_prediction.x

        row_num = data_now.shape[0]
        data_orig_shape = data_now.shape
        data_pred_shape = data_prediction.shape
        assert data_orig_shape == data_pred_shape
        data_now = data_now.view(row_num, -1)
        data_prediction = data_prediction.view(row_num, -1)
        direction = data_prediction - data_now
        t_diff = t_now - t_query
        t_scale = t_diff / t_now
        change_vector = direction * 1.1 * t_scale  # (1.0 - t_query) #t_scale # multiply each column by the corresponding magnitude
        x_tminus1 = data_now + change_vector  # / vectorbridge_magnitude_scale
        if t_query > 1e-3:
            x_tminus1 += torch.randn_like(x_tminus1) * (t_query / vectorbridge_rand_scale)

        x_tminus1 = x_tminus1.reshape(data_orig_shape)  # Use view_as for safety and readability
        data_new.x = x_tminus1
        return data_new



for data in dataloader:
    t = torch.rand(data.x.shape[0]).reshape(-1,1)
    print("input", data, t)
    data_out = model(data, t)
    print("data out", data_out)
    error = torch.sum((data_out.x - data.x)**2)
    break


distance_obj = GraphDistanceAssignment()
def compute_distance(data1, data2):
    original_node_feature_dim = data1.original_node_feature_dim
    original_edge_feature_dim = data1.original_edge_feature_dim
    try:
        original_node_feature_dim = original_node_feature_dim[0]
        original_edge_feature_dim = original_edge_feature_dim[0]
    except:
        pass
    #print("original_node_feature_dim", original_node_feature_dim)
    node_dist = data1.x[data1.node_mask,1:original_node_feature_dim+1] - data2.x[data1.node_mask,1:original_node_feature_dim+1]
    edge_dist = data1.x[data1.edge_mask,1:original_edge_feature_dim+1] - data2.x[data1.edge_mask,1:original_edge_feature_dim+1]
    dist = torch.norm(node_dist) + torch.norm(edge_dist)
    return dist

bridge_obj = VectorBridgeGraph()
pipeline = PipelineVector(node_feature_dim=2, level="DEBUG", degradation_obj=batchify_pyg_transform(degradation_obj), train_obj=train_obj, reconstruction_obj=model, distance_obj=compute_distance, bridge_obj=bridge_obj, pre_trained_path="../pre_trained/qm9_weightsPNA.pt")

print("start training")
dataloader = gen_dataloader(batch_size = 100)
pipeline.train(data=dataloader, epochs=10000)

pipeline.save_all_model_weights("../pre_trained/qm9_weightsPNA.pt")

print("start reconstruction")

dataloader = gen_dataloader(batch_size = 1)
data = next(iter(dataloader))
pipeline.visualize_reconstruction(
    data=data,
    plot_data_func=plot_pyg_graph,
    outfile="images/example8/backward_inference.jpg",
    num=25,
    steps=100,
)




# container:
# create (e.g. takes smiles or molecular graph) - can return 0
# to batch
# distance
# to device
# degrade
# visualize and print

# smiles -> single molecule container -> batched molecule container -> preprocessed/augmented batch -> distance/reconstruction/bridge/degradation
# fils: models, algorithms, utils, containers
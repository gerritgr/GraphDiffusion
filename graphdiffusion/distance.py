import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())

import torch.nn.functional as F

import torch
import torch.nn.functional as F
from graphdiffusion.utils import *


class VectorDistance:
    """
    A class to compute distances between vectors.

    This class supports two types of distances:
    1. L1 (Manhattan) distance
    2. L2 (Euclidean) distance

    Although this class does not inherit from nn.Module like typical PyTorch modules,
    it is designed to function similarly by utilizing the __call__ method.

    Attributes:
        pipeline (object, optional): An optional attribute, not used in the current implementation.
                                    It's reserved for future use if there's a need to integrate this module
                                    within a specific processing pipeline.

    Methods:
        __call__(pipeline, x1, x2, dist_type="L1", *args, **kwargs):
            Computes the distance between two vectors.

    Note:
        The __call__ method is a special method in Python classes. It allows an instance of a class
        to be called as a function. In this case, it is used to compute the distance between two vectors.
    """

    def __init__(self):
        """
        Initializes the VectorDistance class.

        Args:
            pipeline (object, optional): Pipeline object for future use.
        """
        pass

    def __call__(self, x1, x2, dist_type="L1", *args, **kwargs):
        """
        Compute the distance between two vectors based on the specified distance type.

        Args:
            pipeline (object): Pipeline object, unused.
            x1 (Tensor): The first input tensor.
            x2 (Tensor): The second input tensor, must be the same size as x1.
            dist_type (str, optional): The type of distance to compute. Supported values are "L1" and "L2".
                                       Defaults to "L1".

        Returns:
            Tensor: A tensor containing the computed distance.

        Raises:
            ValueError: If the 'dist_type' is not recognized (not "L1" or "L2").
        """
        if dist_type == "L2":
            # Using the built-in mse_loss function for Euclidean distance (L2 norm)
            # The square root of MSE gives the Euclidean distance
            return torch.sqrt(F.mse_loss(x1, x2, reduction="mean"))
        elif dist_type == "L1":
            # Using the built-in l1_loss function for Manhattan distance (L1 norm)
            # It computes the mean absolute error (MAE) between x1 and x2
            return F.l1_loss(x1, x2, reduction="mean")
        else:
            # If the distance type is neither "L1" nor "L2", raise an error
            raise ValueError(f"Unsupported distance type: {dist_type}")

from torch_geometric.nn import SimpleConv, GCNConv, GATConv, GATv2Conv
class SimGCN(torch.nn.Module):
    def __init__(self, input_dim=1):
        super(SimGCN, self).__init__()
        self.input_dict = nn.ModuleDict({str(int(input_dim)): GCNConv(input_dim, 1)})
        self.conv2 = GCNConv(1, 1)
        self.conv3 = GCNConv(1, 1)
        self.conv4 = GCNConv(1, 1)


    def forward(self, x, edge_index):
        input_dim = int(x.shape[1])
        if str(input_dim) not in self.input_dict:
            with DeterministicSeed() as env:
                self.input_dict[str(int(input_dim))] = GCNConv(input_dim, 2)


        conv1 = self.input_dict[str(input_dim)]
        x1 = conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x_nodeembeddings = torch.cat((x1, x2, x3, x4), dim=1)
        x_graphembedding = torch.mean(x_nodeembeddings, dim=0)
        return x_graphembedding, x_nodeembeddings



from torch_geometric.nn import GCN, GIN
class SimGCNAlt(torch.nn.Module):
    def __init__(self, input_dim=1):
        super(SimGCNAlt, self).__init__()
        self.model1 =  GCN(input_dim, 4, 4, 1)
        self.model2 = GCN(1, 4, 4, 1)
        randomize_trainable_params(self.model1, std=3)
        randomize_trainable_params(self.model2, std=3)

    def forward(self, x, edge_index):
        model1 = self.model1 #self.input_dict[str(input_dim)]

        x2 = model1(x*10, edge_index)
        x2 = torch.sin(x2*10)
        x3 = self.model2(x2, edge_index)

        x_nodeembeddings = torch.cat((x2, x3), dim=1) #todo important
        x_graphembedding = torch.mean(x_nodeembeddings, dim=0)
        return x_graphembedding, x_nodeembeddings




import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GraphDistance:
    """
    A class to compute the distance between two graphs using a graph neural network model.

    Attributes:
    - graph_similarity_model (SimGCN): A graph neural network model for computing graph embeddings.

    Methods:
    - __call__(x1, x2, dist_type="L1"): Computes the distance between two graph embeddings.
    """

    def __init__(self, input_dim=4, output_dim=1):
        """
        Initializes the GraphDistance class with a graph similarity model.

        Parameters:
        - input_dim (int): Input dimension of the graph data.
        - output_dim (int): Output dimension of the graph embeddings.
        """
        self.graph_similarity_model = SimGCNAlt() #(input_dim=input_dim, output_dim=output_dim)

    def __call__(self, x1, x2, dist_type="L1"):
        """
        Computes the distance between the embeddings of two graphs.

        Parameters:
        - x1 (Data): The first graph data object.
        - x2 (Data): The second graph data object.
        - dist_type (str): The type of distance to compute ("L1" or "L2").

        Returns:
        - torch.Tensor: The computed distance between the graph embeddings.
        """
        # Input validation
        if not (isinstance(x1, Data) and isinstance(x2, Data)):
            raise ValueError("Both inputs must be PyTorch Geometric Data objects.")
        if not all(hasattr(data, attr) for data in (x1, x2) for attr in ("x", "edge_index")):
            raise ValueError("Input data objects must have 'x' and 'edge_index' attributes.")

        # Obtain graph embeddings

        graph_emb1, _ = self.graph_similarity_model(x1.x, x1.edge_index)
        graph_emb2, _ = self.graph_similarity_model(x2.x, x2.edge_index)


        # Compute distance based on dist_type
        if dist_type == "L1":
            distance = torch.norm(graph_emb1 - graph_emb2, p=1)  # L1 (Manhattan) distance
        elif dist_type == "L2":
            distance = torch.norm(graph_emb1 - graph_emb2, p=2)  # L2 (Euclidean) distance
        else:
            raise ValueError("Unsupported distance type. Choose 'L1' or 'L2'.")

        return distance


from geomloss import SamplesLoss

class GraphDistanceWasserstein:
    """
    A class to compute the distance between two graphs using a graph neural network model.

    Attributes:
    - graph_similarity_model (SimGCN): A graph neural network model for computing graph embeddings.

    Methods:
    - __call__(x1, x2, dist_type="L1"): Computes the distance between two graph embeddings.
    """

    def __init__(self, input_dim=4, output_dim=4):
        """
        Initializes the GraphDistance class with a graph similarity model.

        Parameters:
        - input_dim (int): Input dimension of the graph data.
        - output_dim (int): Output dimension of the graph embeddings.
        """
        self.graph_similarity_model = SimGCNAlt()
        self.loss = SamplesLoss(loss="laplacian", p=2, blur=0.03) 

    def __call__(self, x1, x2, dist_type=None):
        """
        Computes the distance between the embeddings of two graphs.

        Parameters:
        - x1 (Data): The first graph data object.
        - x2 (Data): The second graph data object.
        - dist_type (str): The type of distance to compute ("L1" or "L2").

        Returns:
        - torch.Tensor: The computed distance between the graph embeddings.
        """
        # Input validation
        if not (isinstance(x1, Data) and isinstance(x2, Data)):
            raise ValueError("Both inputs must be PyTorch Geometric Data objects.")
        if not all(hasattr(data, attr) for data in (x1, x2) for attr in ("x", "edge_index")):
            raise ValueError("Input data objects must have 'x' and 'edge_index' attributes.")

        # Obtain graph embeddings
        #self.graph_similarity_model.eval()
        self.graph_similarity_model.train()

        _, node_embeddings1 = self.graph_similarity_model(x1.x, x1.edge_index)
        _, node_embeddings2 = self.graph_similarity_model(x2.x, x2.edge_index)        


        node_embeddings1 = node_embeddings1.view(1, -1, node_embeddings1.shape[-1])  # Reshape for geomloss
        node_embeddings2 = node_embeddings2.view(1, -1, node_embeddings2.shape[-1])  # Reshape for geomloss
        
        distance = self.loss(node_embeddings1, node_embeddings2)


        # TODO: using the wasserstein distance, compute the best mapping/assignment/matching between elements of node_embeddings1 and node_embeddings2 and print it.

        return distance
    


import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_assignment(node_embeddings1, node_embeddings2, return_permutationlist=False, use_cosine=False):
    """
    Computes the best assignment between elements of node_embeddings1 and node_embeddings2,
    both expected to be in NxF format, where N is the number of elements and F is the feature size.
    
    Parameters:
    - node_embeddings1 (torch.Tensor): Embeddings for the first set of nodes, shape (N, F).
    - node_embeddings2 (torch.Tensor): Embeddings for the second set of nodes, shape (M, F).
    - return_permutationlist (bool): If True, returns a permutation list instead of a dictionary.
    - use_cosine (bool): If True, uses cosine similarity for the cost matrix (default: False).
    
    Returns:
    - dict or list: Depending on return_permutationlist, returns a dictionary mapping indices from
      node_embeddings1 to node_embeddings2, or a permutation list for node_embeddings1.
    """
    # Ensure inputs are 2D PyTorch tensors and move them to CPU
    if node_embeddings1.dim() != 2 or node_embeddings2.dim() != 2:
        raise ValueError("Both node_embeddings1 and node_embeddings2 must be 2D tensors.")
    

    node_embeddings1_np = node_embeddings1.detach().cpu().numpy()
    node_embeddings2_np = node_embeddings2.detach().cpu().numpy()

    elem_num = node_embeddings1.shape[0]

    print("input node embeddings for linear assign.: \n", node_embeddings1_np, node_embeddings2_np)
    

    if use_cosine:
        # Compute the cost matrix based on cosine similarity, then invert it since we want to maximize similarity
        cost_matrix = -cosine_similarity(node_embeddings1_np, node_embeddings2_np)
    else:
        # Compute the cost matrix based on squared Euclidean distance
        cost_matrix = np.zeros((elem_num, elem_num))
        for i in range(elem_num):
            for j in range(elem_num):
                cost_matrix[i, j] = np.linalg.norm(node_embeddings1_np[i,:] - node_embeddings2_np[j,:]) #works better without ** 2
                #costs = np.sum((node_embeddings1_np[i,:] - node_embeddings2_np[j,:])**2)
                #cost_matrix[i, j] = costs
                                                   
    print("cost matrix:\n", cost_matrix)
    # Solve the assignment problem
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignment_dict = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    assignment_dict = {value: key for key, value in assignment_dict.items()} # reverse because we apply it to the 2nd argument

    if not return_permutationlist:
        return assignment_dict

    node_permutation = [assignment_dict[i] for i in range(len(assignment_dict))]
    return node_permutation

class GraphDistanceAssignment:
    """
    A class to compute the distance between two graphs using a graph neural network model.

    Attributes:
    - graph_similarity_model (SimGCN): A graph neural network model for computing graph embeddings.

    Methods:
    - __call__(x1, x2, dist_type="L1"): Computes the distance between two graph embeddings.
    """

    def __init__(self, input_dim=4, output_dim=4):
        """
        Initializes the GraphDistance class with a graph similarity model.

        Parameters:
        - input_dim (int): Input dimension of the graph data.
        - output_dim (int): Output dimension of the graph embeddings.
        """
        self.graph_similarity_model = SimGCNAlt()# SimGCN(input_dim=input_dim, output_dim=output_dim)

    def __call__(self, x1, x2, dist_type=None):
        """
        Computes the distance between the embeddings of two graphs.

        Parameters:
        - x1 (Data): The first graph data object.
        - x2 (Data): The second graph data object. This should be the prediction feed through the computational graph. 
        - dist_type (str): The type of distance to compute ("L1" or "L2").

        Returns:
        - torch.Tensor: The computed distance between the graph embeddings.
        """
        # Input validation
        if not (isinstance(x1, Data) and isinstance(x2, Data)):
            raise ValueError("Both inputs must be PyTorch Geometric Data objects.")
        if not all(hasattr(data, attr) for data in (x1, x2) for attr in ("x", "edge_index")):
            raise ValueError("Input data objects must have 'x' and 'edge_index' attributes.")

        # Obtain graph embeddings
        self.graph_similarity_model.eval()
        with torch.no_grad():
            _, node_embeddings1 = self.graph_similarity_model(x1.x, x1.edge_index)
            _, node_embeddings2 = self.graph_similarity_model(x2.x, x2.edge_index)        

            
        node_embeddings1 = torch.cat((x1.x, node_embeddings1), dim=1)
        node_embeddings2 = torch.cat((x2.x, node_embeddings2), dim=1)

        #node_embeddings1 = (node_embeddings1 * 1000).int().float() /1000
        #node_embeddings2 = (node_embeddings2 * 1000).int().float() / 1000


        # TODO multiple calls
        node_permutation = compute_assignment(node_embeddings1, node_embeddings2, return_permutationlist=True)

        node_features = x1.x[node_permutation]

        distance = torch.norm(node_features - x2.x, p=1)

        return distance
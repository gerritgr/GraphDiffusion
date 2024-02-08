import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())

import torch.nn.functional as F

import torch
import torch.nn.functional as F


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





class SimGCN(torch.nn.Module):
    def __init__(self, input_dim=4, output_dim=4):
        super(SimGCN, self).__init__()
        self.input_dict = nn.ModuleDict({
                str(int(input_dim)) : GCNConv(input_dim, output_dim)
        })

        self.conv2 = GCNConv(output_dim, output_dim)
        self.conv3 = GCNConv(output_dim, output_dim)
        self.conv4 = GCNConv(output_dim, output_dim)
        self.out_dim = output_dim

    def forward(self, x, edge_index):
        input_dim = int(x.shape[1])
        if input_dim not in self.input_dict:
            self.input_dict[str(input_dim)] = GCNConv(input_dim, self.out_dim)

        conv1 = self.input_dict[str(input_dim)]
        x1 = conv1(x, edge_index)

        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x_all = torch.cat((x1, x2, x3, x4), dim=1)
        x_all = torch.mean(x_all, dim=0)
        return x_all
    

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
    
    def __init__(self, input_dim=4, output_dim=4):
        """
        Initializes the GraphDistance class with a graph similarity model.

        Parameters:
        - input_dim (int): Input dimension of the graph data.
        - output_dim (int): Output dimension of the graph embeddings.
        """
        self.graph_similarity_model = SimGCN(input_dim=input_dim, output_dim=output_dim)

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
        if not all(hasattr(data, attr) for data in (x1, x2) for attr in ('x', 'edge_index')):
            raise ValueError("Input data objects must have 'x' and 'edge_index' attributes.")
        
        # Obtain graph embeddings
        graph_emb1 = self.graph_similarity_model(x1.x, x1.edge_index)
        graph_emb2 = self.graph_similarity_model(x2.x, x2.edge_index)
        
        # Compute distance based on dist_type
        if dist_type == "L1":
            distance = torch.norm(graph_emb1 - graph_emb2, p=1)  # L1 (Manhattan) distance
        elif dist_type == "L2":
            distance = torch.norm(graph_emb1 - graph_emb2, p=2)  # L2 (Euclidean) distance
        else:
            raise ValueError("Unsupported distance type. Choose 'L1' or 'L2'.")
        
        return distance

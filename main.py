import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100  # Set this to 300 to get better image quality
import seaborn as sns

import networkx as nx
import glob
import random
import os
import traceback
import time
import copy
import pickle
import numpy as np
import math
from tqdm import tqdm
import gzip

from rdkit import Chem
from rdkit.Chem import Draw

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import Sequential as Seq
from torch.nn import Linear as Lin
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    PNA,
    GATv2Conv,
    GraphNorm,
    BatchNorm,
    global_mean_pool,
    global_add_pool
)
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx, degree

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE

print("test")
print("test2")





import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Define a simple default denoiser class based on torch.nn.Module
class DefaultDenoiser(nn.Module):
    def __init__(self, node_feature_dim = 1):
        super(DefaultDenoiser, self).__init__()
        self.node_feature_dim = node_feature_dim


    def forward(self, pipeline, x):
        # Simple denoising logic (this is just a placeholder)
        # Replace with actual denoising logic
        return x

# GraphDiffusionPipeline with a default denoiser
class GraphDiffusionPipeline:
    def __init__(self, batch_size = 1, node_feature_dim = 1, denoiser_object = None):
        # Set to a default denoiser object if none is provided
        self.denoiser_object = denoiser_object or DefaultDenoiser(node_feature_dim = node_feature_dim)
        self.batch_size = batch_size
        self.node_feature_dim = node_feature_dim
        if not callable(self.denoiser_object):
            raise ValueError("denoiser_object must be callable")

    def inference(self):
        # Implementation of the inference method
        pass

    def train(self, data):
        # Convert tensor to DataLoader if necessary
        if isinstance(data, torch.Tensor):
            data = DataLoader(TensorDataset(data))
        # Training logic goes here
        pass

    def denoise(self, data, *args, **kwargs):
        print("denoise")
        # Convert tensor to DataLoader if necessary
        if isinstance(data, torch.Tensor):
            data = DataLoader(TensorDataset(data))
        # Call the denoiser object directly, which invokes its forward() method
        return self.denoiser_object(self, data, *args, **kwargs)

    def forward(self, data):
        # Convert tensor to DataLoader if necessary
        if isinstance(data, torch.Tensor):
            data = DataLoader(TensorDataset(data))
        # Forward logic goes here
        pass

# Example usage
pipeline = GraphDiffusionPipeline()

# Example tensor
example_tensor = torch.randn(10, 3)  # Replace with your actual data

# Using the train method
pipeline.train(example_tensor)

# Using the denoise method
pipeline.denoise(example_tensor)

example_func = lambda x: x

# Using the denoise method
pipeline.denoise(example_func)
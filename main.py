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


# Define a simple default inference class
class DefaultInference(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(DefaultInference, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, noise_to_start, *args, **kwargs):
        # Placeholder inference logic
        return noise_to_start

# Define a simple default forward class
class DefaultForward(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(DefaultForward, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, data, t, *args, **kwargs):
        if t < 1e-7:
            return data
        mean = data * (1-t)
        std_dev = t 
        standard_normal_sample = torch.randn_like(data)
        transformed_sample = mean + std_dev * standard_normal_sample
        return transformed_sample  

# Define a simple default train class
class DefaultTrain():
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(DefaultTrain, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def __call__(self, pipeline, data, epochs, *args, **kwargs):
        # Placeholder training logic
        return data

# Define a simple default bridge class
class DefaultBridge(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(DefaultBridge, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, data_now, data_prediction, t, *args, **kwargs):
        # Placeholder bridge logic
        return data_prediction


# Define a simple default denoiser class based on torch.nn.Module
class DefaultDenoiser(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(DefaultDenoiser, self).__init__()
        self.node_feature_dim = node_feature_dim

    def forward(self, pipeline, data, t, *args, **kwargs):
        # Simple denoising logic (this is just a placeholder)
        # Replace with actual denoising logic
        return data

# GraphDiffusionPipeline with a default denoiser
class GraphDiffusionPipeline:
    def __init__(self, batch_size = 1, node_feature_dim = 1, denoise_obj = None, inference_obj = None, forward_obj = None, train_obj = None, bridge_obj = None):
        self.batch_size = batch_size
        self.node_feature_dim = node_feature_dim

        self.denoise_obj = denoise_obj or DefaultDenoiser(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.inference_obj = inference_obj or DefaultInference(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.forward_obj = forward_obj or DefaultForward(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.train_obj = train_obj or DefaultTrain(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.bridge_obj = bridge_obj or DefaultBridge(node_feature_dim = node_feature_dim, batch_size = batch_size)

        if not callable(self.denoise_obj):
            raise ValueError("denoise_obj must be callable")
        if not callable(self.inference_obj):
            raise ValueError("inference_obj must be callable")
        if not callable(self.forward_obj):
            raise ValueError("forward_obj must be callable")
        if not callable(self.train_obj):
            raise ValueError("train_obj must be callable")
        if not callable(self.bridge_obj):
            raise ValueError("bridge_obj must be callable")

    def brdige(self, data_now, data_prediction, t, *args, **kwargs):
        print("bridge")
        return self.brdige_obj(self, data_now, data_prediction, t, *args, **kwargs)

    def inference(self, data = None, noise_to_start = None, *args, **kwargs):
        print("inference")
        if data is not None and noise_to_start is None:
            raise ValueError("Either data or noise_to_start must be provided")
        return self.inference_obj(self, noise_to_start, *args, **kwargs)

    def train(self, data, epochs=1, *args, **kwargs):
        print("train")
        return self.train_obj(self, data, epochs, *args, **kwargs)

    def denoise(self, data, t, *args, **kwargs):
        print("denoise")
        return self.denoise_obj(self, data, t, *args, **kwargs)

    def forward(self, data, t, *args, **kwargs):
        print("forward")
        return self.forward_obj(self, data, t, *args, **kwargs)
    
    def visualize_foward(self, data, num=100, *args, **kwargs):
        from plotting import create_grid_plot
        print("visualize")
        arrays = list()
        for t in np.linspace(0, 1, num):
            arrays.append(self.forward(data, t))
        return create_grid_plot(arrays, outfile="test.png")

# Example usage
pipeline = GraphDiffusionPipeline()

# Example tensor
example_tensor = torch.randn(10, 3)*10  # Replace with your actual data

# Using the train method
pipeline.train(example_tensor)

# Using the denoise method
pipeline.denoise(example_tensor, 0)

example_func = lambda x: x

# Using the denoise method
pipeline.denoise(example_func, 0)


pipeline.visualize_foward( torch.randn(10) * 5)



print( np.linspace(0, 1, 5))
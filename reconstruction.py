with open('imports.py', 'r') as file:
    exec(file.read())


import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorDenoiser(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size=1, hidden_dim=64, device=None):
        super(VectorDenoiser, self).__init__()
        self.node_feature_dim = node_feature_dim

        # Define MLP layers
        self.fc1 = nn.Linear(node_feature_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        # Second fully connected layer
        self.fc3 = nn.Linear(hidden_dim, node_feature_dim)  # Output layer

    def forward(self, pipeline, data, t, *args, **kwargs):
        data = data.flatten()
        assert(data.numel() == self.node_feature_dim)

        # MLP Logic
        x = F.relu(self.fc1(data))  # Apply first layer with ReLU activation
        x = F.relu(self.fc2(x))     # Apply second layer with ReLU activation
        x = self.fc3(x)             # Apply output layer

        return x

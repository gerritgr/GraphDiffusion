import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imports.py')
with open(file_path, 'r') as file:
    exec(file.read())


import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorDenoiser(nn.Module):
    def __init__(self, node_feature_dim=1, hidden_dim=64, num_layers=8, dropout_rate=0.2, batch_size=1):
        super(VectorDenoiser, self).__init__()
        self.node_feature_dim = node_feature_dim


        # First fully connected layer
        self.fc1 = nn.Linear(node_feature_dim, hidden_dim)

        # Creating hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            layer = nn.Linear(hidden_dim, hidden_dim)
            self.hidden_layers.append(layer)

        # Dropout layers
        self.dropout_layers = nn.Dropout(dropout_rate)

        # Output layer
        self.fc_last = nn.Linear(hidden_dim, node_feature_dim)


    def forward(self, pipeline, data, t, *args, **kwargs):
        data = data.flatten()
        assert(data.numel() == self.node_feature_dim)

        x = F.relu(self.fc1(data))  # Apply first layer with ReLU

        # Apply hidden layers with ReLU and Dropout
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout_layers(x)

        x = self.fc_last(x)  # Apply output layer

        return x

    def save_model(self, pipline, path):
        """
        Save the model weights to the specified path.
        :param path: The file path where the model weights should be saved.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, pipeline, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
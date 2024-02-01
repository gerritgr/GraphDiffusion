import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())


import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorDenoiser(nn.Module):
    def __init__(
        self,
        node_feature_dim=1,
        hidden_dim=256,
        num_layers=8,
        dropout_rate=0.2,
        time_dim=32,
    ):
        # important: config overwrites parameters
        super(VectorDenoiser, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.time_dim = time_dim

        # First fully connected layer
        self.fc1 = nn.Linear(self.node_feature_dim + self.time_dim, hidden_dim)

        # Creating hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            layer = nn.Linear(hidden_dim, hidden_dim)
            self.hidden_layers.append(layer)

        # Dropout layers
        self.dropout_layers = nn.Dropout(dropout_rate)

        # Output layer
        self.fc_last = nn.Linear(hidden_dim, self.node_feature_dim)

    def forward(self, data, t, pipeline, *args, **kwargs):
        # Make sure data has form (batch_size, feature_dim)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        assert data.shape[1] == self.node_feature_dim

        # Add time to data tensor
        # Handle the case where t is a float
        if isinstance(t, float):
            t_tensor = torch.full((data.size(0), 1), t, dtype=data.dtype, device=data.device)
        # Handle the case where t is a tensor of size m
        elif isinstance(t, torch.Tensor) and t.ndim == 1 and t.size(0) == data.size(0):
            t_tensor = t.view(-1, 1)
        else:
            raise ValueError("t must be a float or a 1D tensor of size equal to the number of rows in x")
        # Concatenate t_tensor to x along the last dimension (columns)

        if self.time_dim > 1:
            t_tensor = pipeline.encoding_obj(t_tensor, self.time_dim, add_original=True)
        data = torch.cat((data, t_tensor), dim=1)

        x = F.relu(self.fc1(data))  # Apply first layer with ReLU

        # Apply hidden layers with ReLU and Dropout
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout_layers(x)

        x = self.fc_last(x)  # Apply output layer

        return x

    def save_model(self, pre_trained_path):
        """
        Save the model weights to the specified path.
        :param pre_trained_path: The file path where the model weights should be saved.
        """
        torch.save(self.state_dict(), pre_trained_path)

    def load_model(self, pre_trained_path):
        self.load_state_dict(torch.load(pre_trained_path, map_location=torch.device("cpu")))

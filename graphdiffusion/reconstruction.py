import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imports.py')
with open(file_path, 'r') as file:
    exec(file.read())


import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorDenoiser(nn.Module):
    def __init__(self, node_feature_dim=1, hidden_dim=256, num_layers=8, dropout_rate=0.2, time_dim = 32):
        super(VectorDenoiser, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.time_dim = time_dim


        # First fully connected layer
        self.fc1 = nn.Linear(node_feature_dim+self.time_dim, hidden_dim)

        # Creating hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            layer = nn.Linear(hidden_dim, hidden_dim)
            self.hidden_layers.append(layer)

        # Dropout layers
        self.dropout_layers = nn.Dropout(dropout_rate)

        # Output layer
        self.fc_last = nn.Linear(hidden_dim, node_feature_dim)

    def time_to_pos_emb(self, t, target_dim, add_original = False):
        """
        Converts a tensor of time values into positional embeddings of specified dimensionality.
        
        Args:
        - t (torch.Tensor): A 1D tensor of time values with shape (batch_size,).
        - target_dim (int): The target dimensionality for the positional embeddings.
        
        Returns:
        - torch.Tensor: A tensor of shape (batch_size, target_dim) representing the positional embeddings.
        """
        # Validate input
        t = t.flatten()
        if not (isinstance(t, torch.Tensor) and t.ndim == 1):
            raise ValueError('t must be a 1D tensor')

        # Ensure target_dim is even
        if target_dim % 2 != 0:
            raise ValueError('target_dim must be an even number')
        
        # Get the device from the input tensor
        device = t.device
        
        # Convert t to a 2D tensor with shape (batch_size, 1)
        t_tensor = t.view(-1, 1)
        
        # Convert t_tensor to positional embeddings of dimensionality target_dim
        position = torch.arange(0, target_dim, dtype=torch.float, device=device).unsqueeze(0)
        div_term = torch.exp(position * -(torch.log(torch.tensor(10000.0, device=device)) / target_dim))
        t_pos_emb = torch.zeros_like(t_tensor.expand(-1, target_dim), device=device)
        t_pos_emb[:, 0::2] = torch.sin(t_tensor * div_term[:, 0::2])  # dim 2i
        t_pos_emb[:, 1::2] = torch.cos(t_tensor * div_term[:, 1::2])  # dim 2i+1

        if add_original:
            t_pos_emb[:,-1] = t_tensor.squeeze()
        
        return t_pos_emb
    


    def forward(self, pipeline, data, t, *args, **kwargs):
        # Make sure data has form (batch_size, feature_dim)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        assert(data.shape[1] == self.node_feature_dim)

        # Add time to data tensor
        # Handle the case where t is a float
        if isinstance(t, float):
            t_tensor = torch.full((data.size(0), 1), t, dtype=data.dtype, device=data.device)
        # Handle the case where t is a tensor of size m
        elif isinstance(t, torch.Tensor) and t.ndim == 1 and t.size(0) == data.size(0):
            t_tensor = t.view(-1, 1)
        else:
            raise ValueError('t must be a float or a 1D tensor of size equal to the number of rows in x')
        # Concatenate t_tensor to x along the last dimension (columns)
        
        if self.time_dim > 1:
            t_tensor = self.time_to_pos_emb(t_tensor, self.time_dim, add_original = True)
        data = torch.cat((data, t_tensor), dim=1)


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
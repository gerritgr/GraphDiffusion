import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imports.py')
with open(file_path, 'r') as file:
    exec(file.read())

import torch.nn.functional as F


# Define a simple default forward class
class VectorDistance(nn.Module):
    def __init__(self, node_feature_dim=1, hidden_dim=64, device=None):
        super(VectorDistance, self).__init__()
    
    def forward(self, pipeline, x1, x2, dist_type = "L2", *args, **kwargs):
        if dist_type == "L2":
            # Using the built-in MSEDistance function for Euclidean distance (L2 norm)
            return torch.sqrt(F.mse_loss(x1, x2, reduction='sum'))
        elif dist_type == "L1":
            # Using the built-in L1Distance function for Manhattan distance (L1 norm)
            return F.l1_loss(x1, x2, reduction='sum')
        else:
            raise ValueError(f"Unsupported distance type: {dist_type}")
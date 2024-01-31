import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())


def set_all_seeds(seed=1234):
    """Set the seed for reproducibility in PyTorch, NumPy, and Python random."""

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for Python's random module
    random.seed(seed)


def rand_like_with_seed(data, seed=None):
    """Generate a random tensor of the same shape as data with a seed."""
    if seed is None:
        return torch.randn_like(data, device=data.device)
    old_seed = torch.randint(
        0, 10000, (1,)
    ).item()  # hack to save current state of torch random generator
    set_all_seeds(seed=seed)
    noise = torch.randn_like(data, device=data.device)
    set_all_seeds(seed=old_seed)
    return noise





import torch
import torch.nn as nn

class ModelJoint(nn.Module):
    def __init__(self, models):
        """
        Initialize the ModelJoint with a list of models.

        Args:
            models (list of nn.Module): The list of PyTorch models to be aggregated.
        """
        super(ModelJoint, self).__init__()
        self.models = nn.ModuleList(models)  # ModuleList to properly register all submodules

    def forward(self, x):
        """
        Forward pass sequentially through all models.

        Args:
            x (Tensor): The input tensor to be passed through the model chain.

        Returns:
            Tensor: The output tensor from the last model in the list.
        """
        raise ValueError("Do not call this function directly.")

def create_model_joint(models):
    """
    Create a ModelJoint instance from a list of models.

    Args:
        models (list of nn.Module): The list of PyTorch models to be aggregated.

    Returns:
        ModelJoint: The aggregated model containing all models in the list.
    """
    return ModelJoint(models)

# Example usage:
# model1 = YourModelClass1(...)
# model2 = YourModelClass2(...)
# model_joint = create_model_joint([model1, model2])
# optimizer = torch.optim.Adam(model_joint.parameters(), lr=1e-3)

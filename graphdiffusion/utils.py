import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())


def set_all_seeds(seed=1234):
    """Set the seed for reproducibility in PyTorch, NumPy, and Python random."""
    os.environ["PYTHONHASHSEED"] = "42"

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for Python's random module
    random.seed(seed)


def rand_like_with_seed(data, seed=None):
    """Generate a random tensor of the same shape as data with a seed."""
    if seed is None:
        return torch.randn_like(data, device=data.device)
    old_seed = torch.randint(0, 10000, (1,)).item()  # hack to save current state of torch random generator
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


def unbatch_tensor_list(batched_tensor_list):
    """
    Convert a list of batched tensors into a list of individual tensors.

    Args:
        batched_tensor_list (list of torch.Tensor): A list where each element is a batched tensor.

    Returns:
        list of torch.Tensor: A list where each element is an individual tensor from the batch.
    """
    unbatched_tensor_list = []
    for batched_tensor in batched_tensor_list:
        # Ensure batched_tensor is a tensor
        if not isinstance(batched_tensor, torch.Tensor):
            raise TypeError("All elements of batched_tensor_list should be torch.Tensor")

        # Split the batched tensor along the batch dimension (dim=0) and extend the list
        unbatched_tensor_list.extend(batched_tensor.unbind(dim=0))

    return unbatched_tensor_list


# Example usage:
# model1 = YourModelClass1(...)
# model2 = YourModelClass2(...)
# model_joint = create_model_joint([model1, model2])
# optimizer = torch.optim.Adam(model_joint.parameters(), lr=1e-3)




from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from PIL import Image



def image_to_tensor(image):
    """Convert a PIL image to a PyTorch tensor.

    Args:
        image (PIL.Image): The image to be converted.

    Returns:
        torch.Tensor: The converted PyTorch tensor.
    """
    # Define the image transformation pipeline
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True), transforms.Lambda(lambda t: (t - 0.5) * 2.0)])

    # Apply the transformation pipeline to the image
    return transform(image)


def tensor_to_img(tensor):
    """Convert a PyTorch tensor to a PIL image.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to be converted.

    Returns:
        PIL.Image: The converted PIL image.
    """
    # Define the tensor transformation pipeline
    transform = transforms.Compose([transforms.Lambda(lambda t: (t + 1.0) / 2.0), transforms.Lambda(lambda t: torch.clamp(t, min=0.0, max=1.0)), transforms.ToPILImage()])

    # Apply the transformation pipeline to the tensor
    return transform(tensor)

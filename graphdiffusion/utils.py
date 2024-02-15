import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())


def set_all_seeds(seed=1234):
    """Set the seed for reproducibility in PyTorch, NumPy, and Python random."""
    os.environ["PYTHONHASHSEED"] = str(seed)

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


class DeterministicSeed:
    def __enter__(self, seed=42):
        self.random_seed = torch.randint(0, 100000, (1,)).item()
        set_all_seeds(seed=seed)
        return self  #

    def __exit__(self, exc_type, exc_value, traceback):
        set_all_seeds(seed=self.random_seed)
        return False  # Change to True if you want to suppress exceptions


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


def image_to_tensor(image, img_size):
    """Convert a PIL image to a PyTorch tensor.

    Args:
        image (PIL.Image): The image to be converted.

    Returns:
        torch.Tensor: The converted PyTorch tensor.
    """
    # Define the image transformation pipeline
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((img_size, img_size), antialias=True), transforms.Lambda(lambda t: (t - 0.5) * 2.0)])

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


from torchvision import transforms


def get_simple_image_transform(width, height, convert_to_grayscale=False, convert_to_rgb=False):
    """
    Creates a torchvision transformation pipeline.

    Args:
    - width: Desired output image width.
    - height: Desired output image height.
    - to_gray_scale: Whether to convert images to grayscale.

    Returns:
    - A torchvision.transforms.Compose object.
    """
    transform_list = []

    # Resize the image
    transform_list.append(transforms.Resize((height, width)))

    # Convert image to grayscale if requested, or to RGB if channels == 3 and not converting to grayscale
    if convert_to_grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    elif convert_to_rgb:
        # Convert to RGB by repeating the grayscale channel three times
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    # Convert image to PyTorch tensor
    transform_list.append(transforms.ToTensor())

    # Rescale image values to be between -1 and 1
    transform_list.append(transforms.Lambda(lambda t: (t - 0.5) * 2.0))

    return transforms.Compose(transform_list)


# Example usage with FashionMNIST
# from torchvision.datasets import FashionMNIST
# fashion_mnist_train = FashionMNIST(
#    root="./data",  # Specify the path to store the data
#    train=True,  # Load the training data
#    download=True,  # Download the data if it's not already downloaded
#    transform=get_simple_image_transform(width=16, height=16, convert_to_rgb=True),  # Apply the defined transformations
# )
# for img in fashion_mnist_train:
#    print(img[0].shape)
#    break


import os


def create_path(path):
    """
    Ensure that all directories in the given path exist.

    Parameters:
    - path: The file path for which to ensure directory existence.
    """
    # Extract the directory part of the path
    directory = os.path.dirname(path)

    # If the directory part is not empty, attempt to create the directories
    if directory:
        # The exist_ok=True parameter allows the function to not raise an error if the directory already exists
        os.makedirs(directory, exist_ok=True)
    return path



import torch
from torchvision import transforms

def batchify(single_image_transformation):
    """
    Wrap a single image transformation to make it applicable to batches of images.
    
    Args:
    - single_image_transformation (callable): A transformation function or callable object
      designed to operate on single images.
      
    Returns:
    - Callable: A new transformation function that applies the input transformation to each
      image in a batch of images.
    """
    def transform_batch(batch):
        """
        Apply the wrapped transformation to each image in the batch.
        
        Args:
        - batch (torch.Tensor): A batch of images with shape (B, C, H, W), where B is the
          batch size, C is the number of channels, H is the height, and W is the width.
          
        Returns:
        - torch.Tensor: The batch of images with the transformation applied to each image.
        """
        # Check if the input is a tensor
        if not isinstance(batch, torch.Tensor):
            raise TypeError("The batch must be a torch.Tensor")
        
        # Apply the transformation to each image in the batch
        transformed_batch = torch.stack([single_image_transformation(image) for image in batch])
        
        return transformed_batch
    
    return transform_batch

# Example usage
if __name__ == "__main__":
    # Define a single image transformation
    single_image_transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # Ensuring all images are flipped for demonstration
    ])
    
    # Wrap the single image transformation to make it applicable to batches
    batch_transformation = batchify(single_image_transformation)
    
    # Create a dummy batch of images: 5 images, 3 channels (RGB), 28x28 pixels
    dummy_batch = torch.randn(5, 3, 28, 28)  # Random data simulating a batch of images
    
    # Apply the batch transformation
    transformed_batch = batch_transformation(dummy_batch)
    
    print(f"Original batch shape: {dummy_batch.shape}")
    print(f"Transformed batch shape: {transformed_batch.shape}")

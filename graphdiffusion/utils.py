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
          


################################
################################
# QM 9 utils
################################
################################
          

import torch
from torch_geometric.data import Data

from torch_geometric.utils import to_undirected





def find_edge_attr(data, src, target, undirected=True):
    """
    Find the edge attribute of an edge connecting src to target in a PyG graph.
    If undirected is True, also return the attribute if an edge in the reverse direction exists.

    Parameters:
    - data: A PyTorch Geometric Data object representing the graph.
    - src: The source node id.
    - target: The target node id.
    - undirected: If True, also consider the edge in the reverse direction.

    Returns:
    - The attribute of the edge connecting src to target (or target to src if undirected is True) if it exists, otherwise None.
    """
    # Ensure that src and target are within bounds and edge_attr exists
    if src >= data.num_nodes or target >= data.num_nodes or data.edge_attr is None:
        return None

    # Function to search for an edge and return its attribute
    def search_edge(s, t):
        edge_indices = (data.edge_index[0] == s) & (data.edge_index[1] == t)
        edge_pos = edge_indices.nonzero(as_tuple=True)
        if len(edge_pos[0]) == 0:
            return None
        else:
            return data.edge_attr[edge_pos[0][0]]

    # Search for the edge in the specified direction
    edge_attr_value = search_edge(src, target)
    if edge_attr_value is not None:
        return edge_attr_value

    # If undirected is True and the edge was not found, search in the reverse direction
    if undirected:
        return search_edge(target, src)

    # If no edge is found in either direction, return None
    return None

# Example usage:
# Assuming 'data' is your PyTorch Geometric Data object
# src = 0  # Example source node id
# target = 1  # Example target node id
# Check in both directions
# edge_attr_value = find_edge_attr(data, src, target, undirected=True)
# print(edge_attr_value)

def remove_hydrogens_from_pyg(data):
    from torch_geometric.utils import subgraph
    #data = data.clone()
    data_old = data
    data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    non_hydrogen_mask = data.x[:, 0] == 0
    subset = non_hydrogen_mask.clone().detach()
    new_edge_index, new_edge_attr = subgraph(subset, data.edge_index, data.edge_attr, relabel_nodes = True)
    data.edge_index = new_edge_index
    data.edge_attr = new_edge_attr
    data.x = data.x[non_hydrogen_mask]
    # Copy any additional data
    for key, item in data_old:
        if key not in ['x', 'edge_index', 'edge_attr']:
            data[key] = item
    return data



def inflate_graph(graph, node_indicator=1.0, edge_indicator=-1.0, blank=-1.0, to_undirected=True):
    assert graph.edge_attr is not None
    if abs(node_indicator - edge_indicator) < 0.1:
        raise ValueError("node_indicator and edge_indicator are too close to each other.")


    num_original_nodes = graph.num_nodes
    original_node_feature_dim = graph.x.size(1)
    original_edge_feature_dim = graph.edge_attr.size(1)

    new_node_feature_dim = max(original_node_feature_dim, original_edge_feature_dim) + 1
    new_node_num = num_original_nodes + (num_original_nodes**2 - num_original_nodes) // 2

    new_x = torch.zeros(new_node_num, new_node_feature_dim, device=graph.x.device, dtype=graph.x.dtype) + blank
    new_x[:num_original_nodes, 0] = node_indicator
    new_x[num_original_nodes:, 0] = edge_indicator
    new_x[:num_original_nodes, 1:original_node_feature_dim+1] = graph.x

    edge_list_tuple = []
    new_node_index = num_original_nodes
    for start_i in range(num_original_nodes):
        for target_i in range(num_original_nodes):
            if start_i >= target_i:
                continue
            edge_list_tuple.append((start_i, new_node_index))
            edge_list_tuple.append((new_node_index, target_i))
            edge_attr = find_edge_attr(graph, start_i, target_i)
            if edge_attr is not None:
                new_x[new_node_index, 1:original_edge_feature_dim+1] = edge_attr
            new_node_index += 1

    # Fixing the creation of new_edge_index from edge_list_tuple
    if to_undirected:
        edge_list_tuple = edge_list_tuple + [(j, i) for i, j in edge_list_tuple]
    new_edge_index = torch.tensor(edge_list_tuple, dtype=torch.long).t().contiguous()

    # Create a new Data object with the updated graph structure
    new_graph = Data(x=new_x, edge_index=new_edge_index.to(graph.edge_index.device))
    new_graph.is_inflated = True

    #new_graph.node_mask = new_x[:, 0] == node_indicator
    #new_graph.edge_mask = new_x[:, 0] == edge_indicator
    new_graph.node_mask = torch.isclose(new_x[:, 0], torch.tensor([node_indicator], device=new_x.device), atol=1e-6)
    new_graph.edge_mask = torch.isclose(new_x[:, 0], torch.tensor([edge_indicator], device=new_x.device), atol=1e-6)

    new_graph.original_edge_feature_dim = original_edge_feature_dim
    new_graph.original_node_feature_dim = original_node_feature_dim
    new_graph.num_original_nodes = num_original_nodes

    for key, item in graph:
        if key not in ['x', 'edge_index', 'edge_attr', 'node_mask', 'edge_mask', 'original_edge_feature_dim', 'original_node_feature_dim', 'num_original_nodes']:
            new_graph[key] = item

    return new_graph




def get_all_neighbors(graph, node_id):
    """
    Get all unique neighbors of a node in a PyTorch Geometric graph, considering edges in both directions.

    Parameters:
    - graph: A PyTorch Geometric Data object representing the graph.
    - node_id: The ID of the node for which to find neighbors.

    Returns:
    - A tensor containing the IDs of all unique neighbor nodes.
    """
    # Ensure node_id is within the valid range
    if node_id >= graph.num_nodes:
        raise ValueError("node_id is out of bounds.")

    # Concatenate source and target nodes from edge_index to consider both directions
    all_nodes = torch.cat((graph.edge_index[0], graph.edge_index[1]))
    all_neighbors = torch.cat((graph.edge_index[1], graph.edge_index[0]))

    # Find indices where node_id appears as either source or target
    neighbor_mask = (all_nodes == node_id)

    # Retrieve and return unique neighbor IDs
    neighbors = all_neighbors[neighbor_mask]
    unique_neighbors = torch.unique(neighbors).tolist()

    return unique_neighbors




def reduce_graph(inflated_graph):
    # Check for the inflation flag or specific attributes indicating inflation
    if not hasattr(inflated_graph, 'is_inflated') or not inflated_graph.is_inflated:
        raise ValueError("The input graph does not seem to be inflated.")

    # Retrieve original graph dimensions and node/edge counts
    num_reduced_nodes = inflated_graph.num_original_nodes
    original_node_feature_dim = inflated_graph.original_node_feature_dim
    original_edge_feature_dim = inflated_graph.original_edge_feature_dim

    # Reconstruct original node features from the first part of the inflated graph's x
    reduced_x = inflated_graph.x[:num_reduced_nodes, 1:original_node_feature_dim]

    # Identify nodes and edges based on indicators
    reduced_edge_index = list()
    reduced_edge_attr = list()

    #reduced_edges = inflated_graph.x[inflated_graph.edge_mask, 1:original_edge_feature_dim]

    for i in range(inflated_graph.x.shape[0]):
        if not inflated_graph.edge_mask[i]:
            continue
        edge_i = inflated_graph.x[i, 1:original_edge_feature_dim]
        if edge_i.max() < 0.0:
            continue
        neighbors_of_i = get_all_neighbors(inflated_graph, i)
        assert len(neighbors_of_i) == 2
        reduced_edge_index.append(neighbors_of_i)
        reduced_edge_attr.append(edge_i)

    print("reduced_edge_index", reduced_edge_index)
    print("reduced_edge_attr", reduced_edge_attr)

    reduced_edge_index = torch.tensor(reduced_edge_index, dtype=torch.long).t().contiguous()
    reduced_edge_attr = torch.stack(reduced_edge_attr, dim=0)


    # Reconstruct the original graph Data object
    reduced_graph = Data(x=reduced_x, edge_index=reduced_edge_index, edge_attr=reduced_edge_attr)

    # Copy other attributes from the inflated graph
    for key, item in inflated_graph:
        if key not in ['x', 'edge_index', 'edge_attr', 'is_inflated', 'node_mask', 'edge_mask', 'num_original_nodes', 'original_node_feature_dim', 'original_edge_feature_dim']:
            reduced_graph[key] = item
    
    return reduced_graph

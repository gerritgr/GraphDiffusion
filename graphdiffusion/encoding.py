import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())


def time_to_pos_emb(t, target_dim, add_original=False):
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
        raise ValueError("t must be a 1D tensor")

    # Ensure target_dim is even
    if target_dim % 2 != 0:
        raise ValueError("target_dim must be an even number")

    # Get the device from the input tensor
    device = t.device

    # Convert t to a 2D tensor with shape (batch_size, 1)
    t_tensor = t.view(-1, 1)

    # Convert t_tensor to positional embeddings of dimensionality target_dim
    position = torch.arange(0, target_dim, dtype=torch.float, device=device).unsqueeze(
        0
    )
    div_term = torch.exp(
        position * -(torch.log(torch.tensor(10000.0, device=device)) / target_dim)
    )
    t_pos_emb = torch.zeros_like(t_tensor.expand(-1, target_dim), device=device)
    t_pos_emb[:, 0::2] = torch.sin(t_tensor * div_term[:, 0::2])  # dim 2i
    t_pos_emb[:, 1::2] = torch.cos(t_tensor * div_term[:, 1::2])  # dim 2i+1

    if add_original:
        t_pos_emb[:, -1] = t_tensor.squeeze()

    return t_pos_emb

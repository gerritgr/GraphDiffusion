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
    position = torch.arange(0, target_dim, dtype=torch.float, device=device).unsqueeze(0)
    div_term = torch.exp(position * -(torch.log(torch.tensor(10000.0, device=device)) / target_dim))
    t_pos_emb = torch.zeros_like(t_tensor.expand(-1, target_dim), device=device)
    t_pos_emb[:, 0::2] = torch.sin(t_tensor * div_term[:, 0::2])  # dim 2i
    t_pos_emb[:, 1::2] = torch.cos(t_tensor * div_term[:, 1::2])  # dim 2i+1

    if add_original:
        t_pos_emb[:, -1] = t_tensor.squeeze()

    return t_pos_emb




class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim=32, device=None, embedding_time_scale=1.0):
        super().__init__()
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_time_scale = embedding_time_scale

    def forward(self, time, target_dim=None, add_original=None):
        assert target_dim is None or target_dim == self.dim
        time = time.flatten()
        time = time * self.embedding_time_scale

        half_dim = self.dim * 2 // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if add_original: # TODO you can also fix the dimensions with this (make sure dim can be odd)
            embeddings[:, -1] = time

        return embeddings
    

if __name__ == "__main__":
    # time = torch.tensor([1.0, 1.0, 20.0], dtype=torch.float32)
    time = torch.linspace(0, 1, 30)
    target_dim = 16
    emb_func = SinusoidalPositionEmbeddings(target_dim)
    pos_emb = emb_func(time, target_dim, add_original=True)
    print(pos_emb.shape, pos_emb)

    # Visualize pos_emb as a 2d image
    #plt.figure(figsize=(8, 8))  # Set figure size to be square
    #aspect_ratio = pos_emb.shape[1] / pos_emb.shape[0]  # Width / Height
    plt.imshow(pos_emb, cmap='viridis', aspect=1.3)
    plt.colorbar()
    plt.title("Sinusoidal Position Embeddings")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Time")
    plt.savefig("test_encoding.png", dpi=300)
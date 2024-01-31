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

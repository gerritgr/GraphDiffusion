with open('imports.py', 'r') as file:
    exec(file.read())



# Define a simple default denoiser class based on torch.nn.Module
class DefaultDenoiser(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(DefaultDenoiser, self).__init__()
        self.node_feature_dim = node_feature_dim

    def forward(self, pipeline, data, t, *args, **kwargs):
        # Simple denoising logic (this is just a placeholder)
        # Replace with actual denoising logic
        return data
with open('imports.py', 'r') as file:
    exec(file.read())


# Define a simple default forward class
class DefaultLoss(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size=1, hidden_dim=64, device=None):
        super(DefaultLoss, self).__init__()
    
    def forward(self, pipeline, x1, x2, *args, **kwargs):
        return torch.mean((x1 - x2)**2)
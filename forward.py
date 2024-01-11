with open('imports.py', 'r') as file:
    exec(file.read())


# Define a simple default forward class
class DefaultForward(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(DefaultForward, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, data, t, *args, **kwargs):
        if t < 1e-7:
            return data
        mean = data * (1-t)
        std_dev = t 
        standard_normal_sample = torch.randn_like(data)
        transformed_sample = mean + std_dev * standard_normal_sample
        return transformed_sample  
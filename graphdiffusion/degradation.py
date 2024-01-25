import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imports.py')
with open(file_path, 'r') as file:
    exec(file.read())


# Define a simple default forward class
class VectorAddnoise(nn.Module):
    def __init__(self, node_feature_dim=1):
        super(VectorAddnoise, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, data, t, *args, **kwargs):
        if t < 1e-7:
            return data
        mean = data * (1-t)
        std_dev = t 
        standard_normal_sample = torch.randn_like(data)
        transformed_sample = mean + std_dev * standard_normal_sample
        return transformed_sample  
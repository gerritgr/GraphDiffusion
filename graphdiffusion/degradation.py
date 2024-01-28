import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imports.py')
with open(file_path, 'r') as file:
    exec(file.read())


# Define a simple default forward class
class VectorDegradation(nn.Module):
    def __init__(self, node_feature_dim=1):
        super(VectorDegradation, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, data, t, *args, **kwargs):
        if torch.is_tensor(t):
            batch_dim = data.shape[0]
            t = t.reshape(batch_dim,1)
            if not torch.all((t >= 0) & (t <= 1)):
                raise ValueError("All elements of tensor 't' must be between 0 and 1.")
        elif isinstance(t, float):
            if not (0 <= t <= 1):
                raise ValueError("The float value of 't' must be between 0 and 1.")
        else:
            raise TypeError("'t' must be a float or a tensor.")

        if isinstance(t, float) and t < 1e-7:
            return data

        # If t is a tensor, this operation will be applied element-wise.
        t = t**3.0  # Add scaling
        mean = data * (1 - t)
        std_dev = t**0.5 # you could also just ust t
        standard_normal_sample = torch.randn_like(data)
        transformed_sample = mean + std_dev * standard_normal_sample

        return transformed_sample
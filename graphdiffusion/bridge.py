import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imports.py')
with open(file_path, 'r') as file:
    exec(file.read())




# Define a simple default bridge class
class VectorBridge(nn.Module):
    def __init__(self, node_feature_dim=1):
        super(VectorBridge, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, data_now, data_prediction, t_now, t_query, *args, **kwargs):
        return pipeline.degradation(data_prediction, t_query) # this is not actually a bridge distribution between the two points



class VectorBridgeColdDiffusion(nn.Module):
    def __init__(self, node_feature_dim=1):
        super(VectorBridgeColdDiffusion, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, data_now, data_prediction, t_now, t_query, *args, **kwargs):
        x_0 = data_prediction
        seed = int(random.random()*10000)
        assert(t_now > t_query)
        x_tminus1 = data_now - pipeline.degradation(x_0, t_now, seed=seed) + pipeline.degradation(x_0, t_query, seed=seed)
        return x_tminus1 

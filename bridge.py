with open('imports.py', 'r') as file:
    exec(file.read())




# Define a simple default bridge class
class DefaultBridge(nn.Module):
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(DefaultBridge, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def forward(self, pipeline, data_now, data_prediction, t_now, t_query, *args, **kwargs):
        return pipeline.addnoise(data_prediction, t_query) # this is not actually a bridge distribution between the two points

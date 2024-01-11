
# Define a simple default train class
class DefaultTrain():
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(DefaultTrain, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def __call__(self, pipeline, data, epochs, *args, **kwargs):
        # Placeholder training logic
        return data
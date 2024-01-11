with open('imports.py', 'r') as file:
    exec(file.read())

from denoiser import *
from bridge import *
from train import *
from forward import *
from inference import *

# GraphDiffusionPipeline with a default denoiser
class GraphDiffusionPipeline:
    def __init__(self, batch_size = 1, node_feature_dim = 1, denoise_obj = None, inference_obj = None, forward_obj = None, train_obj = None, bridge_obj = None):
        self.batch_size = batch_size
        self.node_feature_dim = node_feature_dim

        self.denoise_obj = denoise_obj or DefaultDenoiser(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.inference_obj = inference_obj or DefaultInference(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.forward_obj = forward_obj or DefaultForward(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.train_obj = train_obj or DefaultTrain(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.bridge_obj = bridge_obj or DefaultBridge(node_feature_dim = node_feature_dim, batch_size = batch_size)

        if not callable(self.denoise_obj):
            raise ValueError("denoise_obj must be callable")
        if not callable(self.inference_obj):
            raise ValueError("inference_obj must be callable")
        if not callable(self.forward_obj):
            raise ValueError("forward_obj must be callable")
        if not callable(self.train_obj):
            raise ValueError("train_obj must be callable")
        if not callable(self.bridge_obj):
            raise ValueError("bridge_obj must be callable")

    def brdige(self, data_now, data_prediction, t, *args, **kwargs):
        print("bridge")
        return self.brdige_obj(self, data_now, data_prediction, t, *args, **kwargs)

    def inference(self, data = None, noise_to_start = None, *args, **kwargs):
        print("inference")
        if data is not None and noise_to_start is None:
            raise ValueError("Either data or noise_to_start must be provided")
        return self.inference_obj(self, noise_to_start, *args, **kwargs)

    def train(self, data, epochs=1, *args, **kwargs):
        print("train")
        return self.train_obj(self, data, epochs, *args, **kwargs)

    def denoise(self, data, t, *args, **kwargs):
        print("denoise")
        return self.denoise_obj(self, data, t, *args, **kwargs)

    def forward(self, data, t, *args, **kwargs):
        print("forward")
        return self.forward_obj(self, data, t, *args, **kwargs)
    
    def visualize_foward(self, data, outfile = "test.png", num=100, plot_data_func = None):
        from plotting import create_grid_plot
        print("visualize")
        arrays = list()
        for t in np.linspace(0, 1, num):
            arrays.append(self.forward(data, t))
        return create_grid_plot(arrays, outfile=outfile, plot_data_func = plot_data_func)

# Example usage
pipeline = GraphDiffusionPipeline(node_feature_dim=5)

# Example tensor
example_tensor = torch.randn(5)*10  # Replace with your actual data

# Using the train method
pipeline.train(example_tensor)

# Using the denoise method
x = pipeline.denoise(example_tensor, 0)
print(x)



#pipeline.visualize_foward(torch.randn(5) * 5)



print( np.linspace(0, 1, 5))
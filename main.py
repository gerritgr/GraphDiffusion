with open('imports.py', 'r') as file:
    exec(file.read())

from reconstruction import *
from bridge import *
from train import *
from degradation import *
from inference import *
from loss import * 

# VectorPipeline with a default reconstructionr
class VectorPipeline:
    def __init__(self, batch_size = 1, node_feature_dim = 1, device = None, reconstruction_obj = None, inference_obj = None, degradation_obj = None, train_obj = None, bridge_obj = None, distance_obj = None):
        self.batch_size = batch_size
        self.node_feature_dim = node_feature_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reconstruction_obj = reconstruction_obj or VectorDenoiser(node_feature_dim = node_feature_dim, batch_size = batch_size).to(self.device)
        self.inference_obj = inference_obj or VectorInference(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.degradation_obj = degradation_obj or VectorAddnoise(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.train_obj = train_obj or VectorTrain(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.bridge_obj = bridge_obj or VectorBridge(node_feature_dim = node_feature_dim, batch_size = batch_size)
        self.distance_obj = distance_obj or VectorDistance(node_feature_dim = node_feature_dim, batch_size = batch_size)

        if not callable(self.reconstruction_obj):
            raise ValueError("reconstruction_obj must be callable")
        if not callable(self.inference_obj):
            raise ValueError("inference_obj must be callable")
        if not callable(self.degradation_obj):
            raise ValueError("degradation_obj must be callable")
        if not callable(self.train_obj):
            raise ValueError("train_obj must be callable")
        if not callable(self.bridge_obj):
            raise ValueError("bridge_obj must be callable")

    def get_model(self):
        return self.reconstruction_obj.to(self.device)

    def distance(self, x1, x2, *args, **kwargs):
        return self.distance_obj(self, x1, x2, *args, **kwargs)

    def bridge(self, data_now, data_prediction, t_now, t_query, *args, **kwargs):
        print("bridge")
        return self.bridge_obj(self, data_now, data_prediction, t_now, t_query, *args, **kwargs)

    def inference(self, dataloader = None, noise_to_start = None, steps = None, *args, **kwargs): 
        print("inference")
        if data is None and noise_to_start is None:
            raise ValueError("Either data or noise_to_start must be provided")
        return self.inference_obj(self, dataloader, noise_to_start, steps, *args, **kwargs)

    def train(self, data, epochs=100, *args, **kwargs):
        print("train")
        return self.train_obj(self, data, epochs, *args, **kwargs)

    def reconstruction(self, data, t, *args, **kwargs):
        print("reconstruction")
        return self.reconstruction_obj(self, data, t, *args, **kwargs)

    def degradation(self, data, t, *args, **kwargs):
        print("degradation")
        return self.degradation_obj(self, data, t, *args, **kwargs)
    
    def visualize_foward(self, data, outfile = "test.jpg", num=100, plot_data_func = None):
        from plotting import create_grid_plot
        print("visualize")
        arrays = list()
        for t in np.linspace(0, 1, num):
            arrays.append(self.degradation(data, t))
        return create_grid_plot(arrays, outfile=outfile, plot_data_func = plot_data_func)

# Example usage
pipeline = VectorPipeline(node_feature_dim=5)

# Example tensor
example_tensor = torch.randn(5)*10  # Replace with your actual data

# Using the train method
#pipeline.train(example_tensor)

# Using the reconstruction method
x = pipeline.reconstruction(example_tensor, 0)
print("reconstruction", x)

x = pipeline.bridge(data_now = None, data_prediction=example_tensor, t_now=1.0, t_query=0.5)
print("bridge", x)

input_tensor = [5.0 + random.random()*0.01, -5.0 + random.random()*0.01]
random.shuffle(input_tensor)
input_tensor = torch.tensor(input_tensor)
print("input_tensor", input_tensor)
pipeline.visualize_foward(input_tensor)


x1 = [5.0 + random.random()*0.01, -5.0 + random.random()*0.01]
x2 = [5.0 + random.random()*0.01, -5.0 + random.random()*0.01]
print("loss", pipeline.distance(torch.tensor(x1), torch.tensor(x2))) 


pipeline = VectorPipeline(node_feature_dim=2)
tensorlist = [torch.tensor(x1), torch.tensor(x2)]
data = TensorDataset(*tensorlist)
data = DataLoader(tensorlist, batch_size=1, shuffle=True)
pipeline.train(data, epochs=1000)


data0 = pipeline.inference(data)
print("inference", data0)
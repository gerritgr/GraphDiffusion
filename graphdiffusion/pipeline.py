import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'imports.py')
with open(file_path, 'r') as file:
    exec(file.read())

from reconstruction import *
from bridge import *
from train import *
from degradation import *
from inference import *
from distance import * 


# VectorPipeline with a default reconstructionr
class VectorPipeline:
    def __init__(self, pre_trained = None, batch_size = None, node_feature_dim = None, device = None, reconstruction_obj = None, inference_obj = None, degradation_obj = None, train_obj = None, bridge_obj = None, distance_obj = None):
        self.batch_size = batch_size or 1
        self.node_feature_dim = node_feature_dim or 1
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reconstruction_obj = reconstruction_obj or VectorDenoiser(node_feature_dim = node_feature_dim, batch_size = batch_size)

        if pre_trained is not None:
            try:
                # See if load_model is implemented
                self.reconstruction_obj.load_model(self, pre_trained)
            except:
                try:
                    self.reconstruction_obj.load_state_dict(torch.load(pre_trained, map_location=torch.device('cpu')))
                except:
                    print("Could not load pre-trained model.")
            
        self.reconstruction_obj.to(self.device)

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
        if dataloader is None and noise_to_start is None:
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
    
    def visualize_foward(self, data, outfile = "test_forward.jpg", num=100, plot_data_func = None):
        from plotting import create_grid_plot
        print("visualize")
        arrays = list()
        for t in np.linspace(0, 1, num):
            arrays.append(self.degradation(data, t))
        return create_grid_plot(arrays, outfile=outfile, plot_data_func = plot_data_func)

    def visualize_reconstruction(self, data, outfile = "test_backward.jpg", num=25, steps=100, plot_data_func = None):
        from plotting import create_grid_plot
        def split_list(lst, m): #TODO fix
            if m > len(lst):
                raise ValueError("m cannot be greater than the number of elements in the list")
            n = len(lst)
            size = n // m
            extra = n % m
            return [lst[i * size + min(i, extra):(i + 1) * size + min(i + 1, extra)] for i in range(m)]

        print("visualize backward")
        arrays = list()
        backward_steps = list(np.linspace(0., 1, steps))
        backward_steps = split_list(backward_steps, num)
        current_data = self.degradation(data, t=1.0)

        for t in backward_steps:
            current_data = self.inference(noise_to_start=current_data, steps=backward_steps)
            arrays.append(current_data)
        return create_grid_plot(arrays, outfile=outfile, plot_data_func = plot_data_func)


import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())

from reconstruction import *
from bridge import *
from train import *
from degradation import *
from inference import *
from distance import *
from utils import *
from encoding import time_to_pos_emb
from compare_distributions import compare_data_batches


# VectorPipeline with a default reconstructionr
class VectorPipeline:
    def __init__(
        self,
        pre_trained_path=None,
        step_num=100,
        node_feature_dim=None,
        device=None,
        reconstruction_obj=None,
        inference_obj=None,
        degradation_obj=None,
        train_obj=None,
        bridge_obj=None,
        distance_obj=None,
        encoding_obj=None,
        trainable_objects=None,
        **kwargs
    ):
        self.node_feature_dim = node_feature_dim or 1
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_num = step_num
        self.config = kwargs
        self.trainable_objects = trainable_objects

        self.reconstruction_obj = reconstruction_obj or VectorDenoiser(node_feature_dim=node_feature_dim)

        if pre_trained_path is not None:
            try:
                # See if load_model is implemented
                self.reconstruction_obj.load_model(pipeline=self, pre_trained_path=pre_trained_path)
            except:
                try:
                    self.reconstruction_obj.load_state_dict(torch.load(pre_trained_path, map_location=torch.device("cpu")))
                except:
                    print("Could not load pre-trained model.")

        self.reconstruction_obj.to(self.device)

        self.inference_obj = inference_obj or VectorInference()
        self.degradation_obj = degradation_obj or VectorDegradation()
        self.train_obj = train_obj or VectorTrain()
        self.bridge_obj = bridge_obj or VectorBridge()
        self.distance_obj = distance_obj or VectorDistance()
        self.encoding_obj = encoding_obj or time_to_pos_emb

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
        if self.trainable_objects is None:
            return self.reconstruction_obj.to(self.device)
        else:
            assert isinstance(self.trainable_objects, list)
            if self.joint_model:
                return self.joint_model.to(self.device)

        joint_model = create_model_joint(self.trainable_objects)
        self.joint_model = joint_model
        return joint_model.to(self.device)

    def define_trainable_objects(self, reconstruction=True, degragation=False, distance=False, encoding=False):

        # default
        if reconstruction and not degragation and not distance and not encoding:
            self.trainable_objects = None
            return

        self.trainable_objects = list()
        self.joint_model = None
        if reconstruction:
            assert isinstance(self.reconstruction_obj, nn.Module)
            self.trainable_objects.append(self.reconstruction_obj)
        if degragation:
            assert isinstance(self.degradation_obj, nn.Module)
            self.trainable_objects.append(self.degradation_obj)
        if distance:
            assert isinstance(self.distance_obj, nn.Module)
            self.trainable_objects.append(self.distance_obj)
        if encoding:
            assert isinstance(self.encoding_obj, nn.Module)
            self.trainable_objects.append(self.encoding_obj)
        if len(self.trainable_objects) == 0:
            raise ValueError("No trainable objects defined.")

    def distance(self, x1, x2, *args, **kwargs):
        return self.distance_obj(x1, x2, pipeline=self, *args, **kwargs)

    def bridge(self, data_now, data_prediction, t_now, t_query, *args, **kwargs):
        return self.bridge_obj(data_now, data_prediction, t_now, t_query, pipeline=self, *args, **kwargs)

    def inference(self, data=None, noise_to_start=None, steps=None, *args, **kwargs):
        if data is None and noise_to_start is None:
            raise ValueError("Either data or noise_to_start must be provided")
        return self.inference_obj(data=data, noise_to_start=noise_to_start, steps=steps, pipeline=self, *args, **kwargs)
    
    def inference_from_dataloader(self, dataloader, steps=None, *args, **kwargs):
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        generated_data = list()
        for data in dataloader:
            data_new, _ = self.inference(data=data)
            generated_data.append(data_new)
            assert data.shape[0] == data_new.shape[0] # the batch_dim of data_new and data should be the same

        generated_data = unbatch_tensor_list(generated_data)
        assert len(generated_data) == len(dataloader.dataset)
        return generated_data

    def train(self, data, epochs=100, *args, **kwargs):
        return self.train_obj(data=data, epochs=epochs, pipeline=self, *args, **kwargs)

    def reconstruction(self, data, t, *args, **kwargs):
        return self.reconstruction_obj(data=data, t=t, pipeline=self, *args, **kwargs)

    def degradation(self, data, t, *args, **kwargs):
        return self.degradation_obj(data=data, t=t, pipeline=self, *args, **kwargs)

    def visualize_foward(self, data, outfile="test_forward.jpg", num=100, plot_data_func=None):
        from plotting import create_grid_plot

        if isinstance(data, torch.utils.data.DataLoader):
            data = next(iter(data))
        arrays = list()
        for t in np.linspace(0, 1, num):
            arrays.append(self.degradation(data=data, t=t))
        return create_grid_plot(arrays=arrays, outfile=outfile, plot_data_func=plot_data_func)

    def visualize_reconstruction(self, data, outfile="test_backward.jpg", num=25, steps=None, plot_data_func=None):
        from plotting import create_grid_plot

        def split_list(lst, m):  # TODO fix
            if m > len(lst):
                raise ValueError("m cannot be greater than the number of elements in the list")
            n = len(lst)
            size = n // m
            extra = n % m
            final_list = [lst[i * size + min(i, extra) : (i + 1) * size + min(i + 1, extra)] for i in range(m)]
            assert np.sum([len(x) for x in final_list]) == n
            # add additional element to each list
            for i in range(len(final_list) - 1):
                final_list[i].append(final_list[i + 1][0])
            return final_list

        if isinstance(data, torch.utils.data.DataLoader):
            data = next(iter(data))

        steps = steps or self.step_num
        arrays_data = list()
        arrays_projections = list()
        backward_steps = list(np.linspace(1.0, 0, steps))
        backward_steps = split_list(backward_steps, num)
        current_data = self.degradation(data, t=1.0)
        for steps_i in tqdm(backward_steps, total=len(backward_steps), desc="Visualize reconstruction"):
            current_data, current_projection = self.inference(noise_to_start=current_data, steps=steps_i)
            arrays_data.append(current_data)
            arrays_projections.append(current_projection)

        create_grid_plot(arrays_data, outfile=outfile, plot_data_func=plot_data_func)
        return create_grid_plot(
            arrays_projections,
            outfile=outfile.replace(".jpg", "_proj.jpg"),
            plot_data_func=plot_data_func,
        )

    def compare_distribution(self, real_data, generated_data=None, batch_size=100, num_comparisions=10, outfile=None):

        assert isinstance(real_data, torch.utils.data.DataLoader)
        assert generated_data is None or isinstance(generated_data, torch.utils.data.DataLoader)
        assert len(real_data.dataset) >= batch_size*2

        if batch_size < 100:
            print("Warning: batch_size is small, the result may not be accurate.")

        real_dataloader = DataLoader(real_data.dataset, batch_size=batch_size, shuffle=True)
        if generated_data is None:
            generated_data_list = self.inference_from_dataloader(dataloader=real_data)
            generated_dataloader = DataLoader(generated_data_list, batch_size=batch_size, shuffle=True)
        else:
            generated_dataloader = DataLoader(generated_data.dataset, batch_size=batch_size, shuffle=True)

        assert len(real_data.dataset) == len(generated_dataloader.dataset)
        distances_between = list()
        distances_within = list()

        if outfile is not None:
            plt.clf()
            # Calculate number of rows and columns for the grid of subplots
            num_rows = int(np.ceil(np.sqrt(num_comparisions)))
            num_cols = int(np.ceil(num_comparisions / num_rows))

            # Create a figure with num_comparisions axes in a grid
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6 * num_cols, 6 * num_rows))
            axes = axes.flatten()  # Flatten the axes array for easy indexin

        for i in tqdm(range(num_comparisions)):
            # Create iterators for real and generated data loaders
            # Reset dataloader to make sure all batches have the same size and samples are i.i.d. in each comparision
            real_data_iter = iter(real_dataloader)
            generated_data_iter = iter(generated_dataloader)

            real_batch_1 = next(real_data_iter)
            real_batch_2 = next(real_data_iter) # should be disjoint from real_batch_1
            generated_batch = next(generated_data_iter)

            axis = None
            if outfile is not None:
                axis = axes[i] if num_comparisions > 1 else axes

            distance_between, _ = compare_data_batches(real_batch_1, generated_batch, distance_func = self.distance_obj, axis=axis)
            distance_within, _ = compare_data_batches(real_batch_1, real_batch_2, distance_func = self.distance_obj, axis=axis)
            distances_between.append(distance_between)
            distances_within.append(distance_within)

        if outfile is not None:
            plt.tight_layout()
            plt.savefig(outfile)

        return np.nanmean(distances_between), np.nanmean(distances_within), np.nanstd(distances_between), np.nanstd(distances_within)
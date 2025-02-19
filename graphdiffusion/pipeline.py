import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())

from .reconstruction import *
from .bridge import *
from .train import *
from .degradation import *
from .inference import *
from .distance import *
from .utils import *
from .config import *
from .compare_distributions import compare_data_batches
from .plotting import *


class PipelineBase:
    def __init__(
        self,
        node_feature_dim,
        device,
        reconstruction_obj,
        inference_obj,
        degradation_obj,
        train_obj,
        bridge_obj,
        distance_obj,
        trainable_objects,
        pre_trained_path,
        logger=None,
        clamp_inference=None,
        preprocess_batch = None,
        postprocess_batch = None,
        **kwargs,
    ):
        self.device = device
        self.node_feature_dim = node_feature_dim
        self.reconstruction_obj = reconstruction_obj
        self.inference_obj = inference_obj
        self.degradation_obj = degradation_obj
        self.train_obj = train_obj
        self.bridge_obj = bridge_obj
        self.distance_obj = distance_obj
        self.trainable_objects = trainable_objects
        self.clamp_inference = clamp_inference
        self.preprocess_batch = preprocess_batch
        self.postprocess_batch = postprocess_batch

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
        if not callable(self.distance_obj):
            raise ValueError("distance_obj must be callable")
        assert isinstance(self.node_feature_dim, int)

        if preprocess_batch and not callable(self.preprocess_batch):
            raise ValueError("preprocess_batch must be callable")
        if postprocess_batch and not callable(self.postprocess_batch):
            raise ValueError("postprocess_batch must be callable")

        if trainable_objects is None:
            assert isinstance(reconstruction_obj, nn.Module)
        else:
            for obj in trainable_objects:
                assert isinstance(obj, nn.Module)

        self.logger = logger
        if self.logger is None:
            from loguru import logger

            self.logger = logger
            self.logger.debug("Using default loguru logger.")

        if pre_trained_path is not None:
            self.load_all_model_weights(pre_trained_path)
            # try:
            #    # See if load_model is implemented
            #    self.reconstruction_obj.load_model(pre_trained_path)
            # except:
            #    try:
            #        self.reconstruction_obj.load_state_dict(torch.load(pre_trained_path, map_location=torch.device("cpu")))
            #    except:
            #        print("Could not load pre-trained model.")

        try:
            self.config["node_feature_dim"] = node_feature_dim
            self.config["device"] = device
            self.config["pre_trained_path"] = pre_trained_path
            for key, value in kwargs.items():
                self.config[key] = value
        except:
            self.config = get_config()
            self.config["node_feature_dim"] = node_feature_dim
            self.config["device"] = device
            self.config["pre_trained_path"] = pre_trained_path
            for key, value in kwargs.items():
                self.config[key] = value

        try:
            if "level" not in self.config or not isinstance(self.config["level"], str):
                self.logger.debug("Set logging level to INFO.")
                self.logger.remove()
                self.logger.add(sys.stderr, level="INFO")
            else:
                self.logger.debug(f"Set logging level to {self.config['level']}.")
                self.logger.remove()
                self.logger.add(sys.stderr, level=self.config["level"])
        except:
            pass

    def preprocess(self, batch):
        if self.preprocess_batch is not None:
            batch = self.preprocess_batch(batch)
        return batch
    
    def postprocess(self, batch):
        if self.postprocess_batch is not None:
            batch = self.postprocess_batch(batch)
        return batch

    def get_model(self):  # TODO does not work with saving loading
        if self.trainable_objects is None:
            return self.reconstruction_obj.to(self.device)
        else:
            assert isinstance(self.trainable_objects, list)
            try:
                return self.joint_model.to(self.device)
            except:
                pass

        joint_model = create_model_joint(self.trainable_objects)
        self.joint_model = joint_model
        return joint_model.to(self.device)

    def info(self, *args, **kwargs):
        return self.logger.info(" ".join([str(x) for x in args]), **kwargs)

    def debug(self, *args, **kwargs):
        return self.logger.debug(" ".join([str(x) for x in args]), **kwargs)

    def warning(self, *args, **kwargs):
        return self.logger.warning(" ".join([str(x) for x in args]), **kwargs)

    def warn(self, *args, **kwargs):
        return self.logger.warning(" ".join([str(x) for x in args]), **kwargs)

    def success(self, *args, **kwargs):
        return self.logger.success(" ".join([str(x) for x in args]), **kwargs)

    def define_trainable_objects(self, reconstruction=True, degragation=False, distance=False):

        # default
        if reconstruction and not degragation and not distance:
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
        if len(self.trainable_objects) == 0:
            raise ValueError("No trainable objects defined.")

    def distance(self, x1, x2, **kwargs):
        params = get_params(self.distance_obj, self.config, kwargs)
        return self.distance_obj(x1, x2, **params)

    def bridge(self, data_now, data_prediction, t_now, t_query, clamp_bridge=None, **kwargs):
        params = get_params(self.bridge_obj, self.config, kwargs)
        results = self.bridge_obj(data_now, data_prediction, t_now, t_query, self, **params)
        if clamp_bridge is None and "clamp_bridge" in self.config:
            clamp_bridge = self.config["clamp_bridge"]
        if clamp_bridge is not None:
            results = torch.clamp(results, *clamp_bridge)  # TODO make config
        return results

    def inference(self, data, noise_to_start, steps=None, **kwargs):
        if data is None and noise_to_start is None:
            raise ValueError("Either data or noise_to_start must be provided.")
        steps = steps or self.config["step_num"]
        params = get_params(self.inference_obj, self.config, kwargs)
        result = self.inference_obj(data, self, noise_to_start, steps, **params)
        if not isinstance(result, tuple) or len(result) != 2 or result[0] is None or result[1] is None:
            raise ValueError("Inference results must be a tuple with exactly two non-None elements.")
        return result

    def inference_from_dataloader(self, dataloader, **kwargs):
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        # params = get_params(self.inference_obj, self.config, kwargs)
        generated_data = list()
        for data in dataloader:
            data_new, _ = self.inference(data=data, noise_to_start=None, steps=self.config.step_num, **kwargs)
            generated_data.append(data_new)
            assert data.shape[0] == data_new.shape[0]  # the batch_dim of data_new and data should be the same

        generated_data = unbatch_tensor_list(generated_data)
        assert len(generated_data) == len(dataloader.dataset)
        return generated_data

    def train(self, data, **kwargs):
        params = get_params(self.train_obj, self.config, kwargs)
        if "TEST_MODUS_WITH_REDUCED_TRAINING" in globals() and TEST_MODUS_WITH_REDUCED_TRAINING:
            params["epochs"] = 5
        return self.train_obj(data, self, **params)

    def reconstruction(self, data, t, **kwargs):
        params = get_params(self.reconstruction_obj, self.config, kwargs)
        return self.reconstruction_obj(data, t, self, **params)

    def degradation(self, data, t, **kwargs):
        params = get_params(self.degradation_obj, self.config, kwargs)
        return self.degradation_obj(data, t, self, **params)

    def visualize_foward(self, data, outfile, num, plot_data_func):
        from .plotting import create_grid_plot

        plt.close()

        if isinstance(data, torch.utils.data.DataLoader):
            data = next(iter(data))
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]
        if isinstance(data, torch.Tensor) and data.dim() == 1:
            data = data.view(1, -1)

        arrays = list()
        for t in np.linspace(0, 1, num):
            arrays.append(self.degradation(data, t))
        return create_grid_plot(arrays=arrays, outfile=outfile, plot_data_func=plot_data_func)

    def visualize_reconstruction(self, data, outfile, outfile_projection, num, steps, plot_data_func):
        from .plotting import create_grid_plot

        plt.close()

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
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]
            data = self.preprocess(data)

        arrays_data = list()
        arrays_projections = list()
        backward_steps = list(np.linspace(1.0, 0, steps))
        backward_steps = split_list(backward_steps, num)
        current_data = self.degradation(data, t=1.0)
        for steps_i in tqdm(backward_steps, total=len(backward_steps), desc="Visualize reconstruction"):
            assert current_data is not None
            current_data, current_projection = self.inference(data=None, noise_to_start=current_data, steps=steps_i)
            arrays_data.append(current_data)
            arrays_projections.append(current_projection)

        arrays_data = [self.postprocess(data) for data in arrays_data]
        create_grid_plot(arrays_data, outfile=outfile, plot_data_func=plot_data_func)
        return create_grid_plot(
            arrays_projections,
            outfile=outfile_projection,
            plot_data_func=plot_data_func,
        )

    # def save_reconstruction_model(self, model_path):
    #    try:
    #        self.reconstruction_obj.save_model(model_path)
    #    except:
    #        model = self.reconstruction_obj
    #        torch.save(model.state_dict(), model_path)

    def info_to_str(self):
        config_local = self.config.copy()
        model = self.get_model()
        indent = "      "
        model_num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        config_local["Number of trainable parameters"] = model_num_params
        if isinstance(self.reconstruction_obj, nn.Module):
            config_local["Reconstruction model"] = str(self.reconstruction_obj).replace("\n", "\n" + indent)
        if isinstance(self.degradation_obj, nn.Module):
            config_local["Degradation model"] = str(self.degradation_obj).replace("\n", "\n" + indent)
        if isinstance(self.distance_obj, nn.Module):
            config_local["Distance model"] = str(self.distance_obj).replace("\n", "\n" + indent)

        config = "\n".join([f"{indent}{key}: {value}" for key, value in config_local.items()])
        return f"Pipeline with the following configuration:\n{config}"

    def save_all_model_weights(self, model_path, print_process=True, save_optimizer_state=True):
        models = dict()
        if isinstance(self.reconstruction_obj, nn.Module):
            models["reconstruction_obj"] = self.reconstruction_obj
        if isinstance(self.degradation_obj, nn.Module):
            models["degradation_obj"] = self.degradation_obj
        if isinstance(self.distance_obj, nn.Module):
            models["distance_obj"] = self.distance_obj
        if save_optimizer_state:
            try:
                models["optimizer"] = self.optimizer
            except:
                self.warning("Warning: optimizer not saved.")

        if print_process:
            self.info(f"Save models {models.keys()} to: {model_path}.")

        model_state_dicts = {name: model.state_dict() for name, model in models.items()}
        torch.save(model_state_dicts, model_path)

    def load_all_model_weights(self, model_path, print_process=True, load_optimizer_state=True):
        if not os.path.exists(model_path):
            self.warning(f"Model file {model_path} does not exist. Cannot load weights.")
            return

        # Load the saved model state dictionaries
        model_state_dicts = torch.load(model_path)
        model_list = list()

        # Assuming self has the model attributes
        if "reconstruction_obj" in model_state_dicts and isinstance(self.reconstruction_obj, nn.Module):
            self.reconstruction_obj.load_state_dict(model_state_dicts["reconstruction_obj"])
            model_list.append("reconstruction_obj")

        if "degradation_obj" in model_state_dicts and isinstance(self.degradation_obj, nn.Module):
            self.degradation_obj.load_state_dict(model_state_dicts["degradation_obj"])
            model_list.append("degradation_obj")

        if "distance_obj" in model_state_dicts and isinstance(self.distance_obj, nn.Module):
            self.distance_obj.load_state_dict(model_state_dicts["distance_obj"])
            model_list.append("distance_obj")

        if load_optimizer_state:
            if "optimizer" in model_state_dicts:
                self.optimizer_state_dict = model_state_dicts["optimizer"]
                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                # optimizer.load_state_dict(model_state_dicts["optimizer"])
                model_list.append("optimizer")
            else:
                self.info("Warning: optimizer state not loaded.")

        if print_process:
            self.info(f"Loaded models from: {model_path}, loaded: {str(model_list)}.")

    def compare_distribution(self, real_data, generated_data, batch_size, num_comparisions, outfile, max_plot, compare_data_batches_func, **kwargs):

        assert isinstance(real_data, torch.utils.data.DataLoader)
        assert generated_data is None or isinstance(generated_data, torch.utils.data.DataLoader)
        assert len(real_data.dataset) >= batch_size * 2

        if batch_size < 100:
            self.info("Warning: batch_size is small, the result may not be accurate.")

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
            num_plots = 2 * num_comparisions  # Double the number of axes
            num_plots = min(num_plots, max_plot)
            num_rows = int(np.ceil(np.sqrt(num_plots)))
            num_cols = int(np.ceil(num_plots / num_rows))

            # Create a figure with num_plots axes in a grid
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6 * num_cols, 6 * num_rows))
            axes = axes.flatten()  # Flatten the axes array for easy indexing

        for i in tqdm(range(num_comparisions), desc="Compare distributions:"):
            # Create iterators for real and generated data loaders
            # Reset dataloader to make sure all batches have the same size and samples are i.i.d. in each comparision
            real_data_iter = iter(real_dataloader)
            generated_data_iter = iter(generated_dataloader)

            real_batch_1 = next(real_data_iter)
            real_batch_2 = next(real_data_iter)  # should be disjoint from real_batch_1
            generated_batch = next(generated_data_iter)

            axis_between = None
            axis_within = None
            if outfile is not None:
                idx = 2 * i  # Index for the first axis of the pair
                if idx < num_plots:
                    axis_between = axes[idx]
                    axis_within = axes[idx + 1]
                    axis_between.set_title("Optimal transport map between real and generated data")
                    axis_within.set_title("Optimal transport map between batches real data")

            distance_between, _ = compare_data_batches_func(real_batch_1, generated_batch, distance_func=self.distance, axis=axis_between, **kwargs)
            distance_within, _ = compare_data_batches_func(real_batch_1, real_batch_2, distance_func=self.distance, axis=axis_within, color_generated="orange", **kwargs)
            distances_between.append(distance_between)
            distances_within.append(distance_within)

        if outfile is not None:
            plt.tight_layout()
            plt.savefig(create_path(outfile))

        result_dict = {
            "mean distance (real vs generated)": np.nanmean(distances_between),
            "mean distance (real vs real)": np.nanmean(distances_within),
            "std distance (real vs generated)": np.nanstd(distances_between),
            "std distance (real vs real)": np.nanstd(distances_within),
        }
        return result_dict


        
# Vector means any 1D data, e.g. time series, 1D vectors, images that are flattened, etc.
class PipelineVector(PipelineBase):
    def __init__(
        self,
        node_feature_dim=None,
        device=None,
        reconstruction_obj=None,
        inference_obj=None,
        degradation_obj=None,
        train_obj=None,
        bridge_obj=None,
        distance_obj=None,
        trainable_objects=None,
        pre_trained_path=None,
        **kwargs,
    ):

        self.config = get_config()
        self.config.node_feature_dim = node_feature_dim or 1
        node_feature_dim = node_feature_dim or 1
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        reconstruction_obj = reconstruction_obj or VectorDenoiser(**get_params(VectorDenoiser.__init__, self.config, kwargs))
        inference_obj = inference_obj or VectorInference()
        degradation_obj = degradation_obj or  VectorDegradationDDPM(**get_params(VectorDegradationDDPM.__init__, self.config, kwargs))   #VectorDegradation(**get_params(VectorDegradation.__init__, self.config, kwargs))
        train_obj = train_obj or VectorTrain()
        bridge_obj = bridge_obj or VectorBridge()
        distance_obj = distance_obj or VectorDistance()

        super().__init__(node_feature_dim, device, reconstruction_obj, inference_obj, degradation_obj, train_obj, bridge_obj, distance_obj, trainable_objects, pre_trained_path, **kwargs)

        for key, value in kwargs.items():
            self.config[key] = value

        self.info(self.info_to_str())

    def visualize_foward(self, data, outfile, num=100, plot_data_func=None):
        if plot_data_func is None:
            if self.config.node_feature_dim == 2:
                plot_data_func = plot_2darray_on_axis
            else:
                plot_data_func = plot_array_on_axis
        super().visualize_foward(data, outfile, num, plot_data_func)

    def visualize_reconstruction(self, data, outfile, outfile_projection=None, num=None, steps=None, plot_data_func=None):
        if outfile_projection is None:
            for ending in [".jpg", ".pdf", ".png", ".jpeg", ".svg"]:
                if ending in outfile:
                    outfile_projection = outfile.replace(ending, "") + "_proj" + ending
                    break
        if num is None:
            num = 25
        if steps is None:
            steps = self.config["step_num"] or 100
        if plot_data_func is None:
            if self.config.node_feature_dim == 2:
                plot_data_func = plot_2darray_on_axis
            else:
                plot_data_func = plot_array_on_axis
        super().visualize_reconstruction(data, outfile, outfile_projection, num, steps, plot_data_func)

    def compare_distribution(self, real_data, generated_data=None, batch_size=200, num_comparisions=32, outfile=None, max_plot=36, **kwargs):
        if outfile is not None and self.config.node_feature_dim != 2:
            self.info("Warning: compare_distribution only shows 2 dimensions.")
        return super().compare_distribution(real_data, generated_data, batch_size, num_comparisions, outfile, max_plot, compare_data_batches, **kwargs)

    @staticmethod
    def create_from_dataloader(dataloader, **kwargs):
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        data = next(iter(dataloader))
        if isinstance(data, list) or isinstance(data, tuple):
            data = data[0]
        assert isinstance(data, torch.Tensor)
        assert len(data.shape) == 3 or len(data.shape) == 4
        if len(data.shape) == 3:
            channels = 1
            img_width = data.shape[1]
            img_height = data.shape[2]
        else:
            channels = data.shape[1]
            img_width = data.shape[2]
            img_height = data.shape[3]

        pipeline = PipelineImage(channels=channels, img_width=img_width, img_height=img_height, **kwargs)
        pipeline.debug("Created pipeline from dataloader.")
        return pipeline


class PipelineImage(PipelineBase):
    def __init__(
        self,
        channels=3,
        img_width=16,
        img_height=16,
        device=None,
        reconstruction_obj=None,
        inference_obj=None,
        degradation_obj=None,
        train_obj=None,
        bridge_obj=None,
        distance_obj=None,
        trainable_objects=None,
        pre_trained_path=None,
        **kwargs,
    ):

        self.config = get_config()
        assert isinstance(channels, int) and img_width > 0
        assert isinstance(img_height, int) and img_width > 0
        assert isinstance(img_width, int) and img_width > 0
        assert img_height == img_width  # currently only support for square images

        self.channels = channels
        self.img_width = img_width
        self.img_height = img_height

        node_feature_dim = img_width * img_height * channels
        self.config.node_feature_dim = node_feature_dim
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        reconstruction_obj = reconstruction_obj or ImageReconstruction(dim=img_width, channels=channels, device=device)
        inference_obj = inference_obj or VectorInference()  # can stay
        degradation_obj = degradation_obj or VectorDegradationDDPM(**get_params(VectorDegradationDDPM.__init__, self.config, kwargs))
        train_obj = train_obj or VectorTrain()  # can stay
        bridge_obj = bridge_obj or VectorBridgeDDPM()  # can stay
        distance_obj = distance_obj or VectorDistance()  # can stay

        super().__init__(node_feature_dim, device, reconstruction_obj, inference_obj, degradation_obj, train_obj, bridge_obj, distance_obj, trainable_objects, pre_trained_path)

        for key, value in kwargs.items():
            self.config[key] = value

        self.info(self.info_to_str())

    def visualize_foward(self, data, outfile, num=100, plot_data_func=None):
        if plot_data_func is None:
            plot_data_func = plot_image_on_axis
        super().visualize_foward(data, outfile, num, plot_data_func)

    def visualize_reconstruction(self, data, outfile, outfile_projection=None, num=None, steps=None, plot_data_func=None):
        if outfile_projection is None:
            for ending in [".jpg", ".pdf", ".png", ".jpeg", ".svg"]:
                if ending in outfile:
                    outfile_projection = outfile.replace(ending, "") + "_proj" + ending
                    break
        if num is None:
            num = 25
        if steps is None:
            steps = self.config["step_num"] or 100
        if plot_data_func is None:
            plot_data_func = plot_image_on_axis
        super().visualize_reconstruction(data, outfile, outfile_projection, num, steps, plot_data_func)

    def compare_distribution(self, real_data, generated_data=None, batch_size=200, num_comparisions=32, outfile=None, max_plot=32, **kwargs):
        if outfile is not None and self.config.node_feature_dim != 2:
            self.info("Warning: compare_distribution only shows 2 dimensions.")
        return super().compare_distribution(real_data, generated_data, batch_size, num_comparisions, outfile, max_plot, compare_data_batches, **kwargs)

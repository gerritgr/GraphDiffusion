import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())

from .utils import *
from .pipeline import *


class VectorDegradation:
    """
    A PyTorch module for simulating the degradation of vectors. It modifies the input data by a degradation factor 't'
    and applies scaling to 't' and the standard deviation before degradation.

    The degradation is modelled as a linear transformation of the input data,
    scaling it by (1-t) and adding a noise component. The noise follows a standard normal distribution
    and is scaled by t raised to the power of std_dev_scaling_factor.

    Attributes:
        pipeline (object): A pipeline object used for processing. It is not utilized in the current implementation
                           but is included for future compatibility with complex workflows.
        time_scaling_factor (float): The exponent to which the degradation factor 't' is raised. Defaults to 3.0.
        std_dev_scaling_factor (float): The exponent to which 't' is raised to compute the standard deviation
                                        for the noise component. Defaults to 0.5.

    Methods:
        forward(pipeline, data, t, seed=None, *args, **kwargs):
            Applies the degradation transformation to the input data.
    """

    def __init__(self, time_scaling_factor=3.0, std_dev_scaling_factor=0.5):
        """
        Initializes the VectorDegradation module. The pipeline parameter is stored but not currently used.
        The time_scaling_factor parameter is used to control the non-linear scaling of the degradation factor 't',
        and std_dev_scaling_factor is used to control the scaling of the standard deviation of the noise.

        Args:
            pipeline (object): An object representing the processing pipeline for future use.
            time_scaling_factor (float, optional): The exponent to which the degradation factor 't' is raised.
                                              Defaults to 3.0. Large values mean that the structure remains intact for longer.
            std_dev_scaling_factor (float, optional): The exponent to which 't' is raised to compute the standard deviation
                                                      for the noise component. Defaults to 0.5. Small values mean that the standard deviation converges faster to 1.
        """
        # super(VectorDegradation, self).__init__()

        self.time_scaling_factor = time_scaling_factor
        self.std_dev_scaling_factor = std_dev_scaling_factor
        assert self.time_scaling_factor > 0, "The time scaling factor must be positive."
        assert self.std_dev_scaling_factor > 0, "The standard deviation scaling factor must be positive."

    def __call__(self, data, t, pipeline, seed=None, **kwargs):
        """
        Apply the degradation process to the input data based on the degradation factor 't',
        the scaling factor, and the standard deviation scaling factor.

        Args:
            pipeline (object): Pipeline object, not utilized in the current implementation.
            data (Tensor): The input tensor to be degraded.
            t (float or Tensor): The degradation factor. If it's a tensor, it should have the same batch size as the data.
                                 Each element of 't' should be in the range [0, 1].
            seed (int, optional): A seed for random number generation to ensure reproducibility.

        Returns:
            Tensor: The degraded version of the input data.

        Raises:
            ValueError: If 't' has values outside the range [0, 1].
            TypeError: If 't' is not a float or a tensor.
        """
        if torch.is_tensor(t):
            batch_dim = data.shape[0]
            t = t.reshape(batch_dim, 1)
            if not torch.all((t >= 0) & (t <= 1)):
                raise ValueError("All elements of tensor 't' must be between 0 and 1.")
        elif isinstance(t, float):
            if not (0 <= t <= 1):
                raise ValueError("The float value of 't' must be between 0 and 1.")
        else:
            raise TypeError("'t' must be a float or a tensor.")

        if isinstance(t, float) and t < 1e-7:
            # If 't' is very small, return the data unmodified as the degradation is negligible.
            return data

        data_orig_shape = data.shape
        data = data.view(data.shape[0], -1)

        # Transform 't' by raising it to the power of the scaling factor.
        t = t**self.time_scaling_factor
        # Compute the mean of the degraded data.
        mean = data * (1 - t)
        # Compute the standard deviation for the noise component based on 't' and the std_dev_scaling_factor.
        std_dev = t**self.std_dev_scaling_factor

        # Generate a sample from the standard normal distribution.
        standard_normal_sample = rand_like_with_seed(data, seed=seed)

        # Apply the degradation transformation.
        transformed_sample = mean + std_dev * standard_normal_sample

        transformed_sample = transformed_sample.view(data_orig_shape)
        return transformed_sample


class VectorDegradationHighVariance:
    def __init__(self, time_scaling_factor=1.0, std_dev_max=2.0):
        # super(VectorDegradationHighVariance, self).__init__()

        self.time_scaling_factor = time_scaling_factor
        self.std_dev_max = std_dev_max
        assert self.time_scaling_factor > 0, "The time scaling factor must be positive."
        assert self.std_dev_max > 0, "The standard deviation scaling factor must be positive."

    def __call__(self, data, t, pipeline, seed=None, **kwargs):

        if torch.is_tensor(t):
            batch_dim = data.shape[0]
            t = t.reshape(batch_dim, 1)
            if not torch.all((t >= 0) & (t <= 1)):
                raise ValueError("All elements of tensor 't' must be between 0 and 1.")
        elif isinstance(t, float):
            if not (0 <= t <= 1):
                raise ValueError("The float value of 't' must be between 0 and 1.")
        else:
            raise TypeError("'t' must be a float or a tensor.")

        if isinstance(t, float) and t < 1e-7:
            # If 't' is very small, return the data unmodified as the degradation is negligible.
            return data
        elif isinstance(t, float):
            t = torch.tensor(t, device=data.device)

        # Change view to get rid of the extra dimensions
        batch_size = data.shape[0]
        old_shape = data.shape
        data = data.view(batch_size, -1)

        # Transform 't' by raising it to the power of the scaling factor.
        t = t**self.time_scaling_factor
        # Compute the mean of the degraded data.
        mean = data * (1 - t)
        # Compute the standard deviation for the noise component based on 't' and the std_dev_max.
        # std_dev  = Piecewise[{{x*2*5, x < 0.5}, {-2*(5-1)*x  + (5 - (-2*(5-1)*0.5)), x >= 0.5}}] # assuming 5 is the max std
        mask = t < 0.5
        std_dev = torch.zeros_like(t)
        std_dev[mask] = t[mask] * 2 * self.std_dev_max
        std_dev[~mask] = -2 * (self.std_dev_max - 1) * t[~mask] + (self.std_dev_max - (-2 * (self.std_dev_max - 1) * 0.5))

        # Generate a sample from the standard normal distribution.
        standard_normal_sample = rand_like_with_seed(data, seed=seed)

        # Apply the degradation transformation.
        transformed_sample = mean + std_dev * standard_normal_sample

        transformed_sample = transformed_sample.view(*old_shape)
        return transformed_sample


class VectorDegradationDDPM(nn.Module):
    def __init__(self):
        """
        Initializes the VectorDegradationDDPM module. The pipeline parameter is stored but not currently used.

        Args:
            pipeline (object, optional): An object representing the processing pipeline, unused.
        """
        super(VectorDegradationDDPM, self).__init__()

    @staticmethod
    def generate_schedule(ddpm_start=0.0001, ddpm_end=0.04, step_num=100):
        """
        Generates a schedule of beta values for a forward process.

        Args:
            start (float): The starting value for the beta values. Default is 0.0001.
            end (float): The ending value for the beta values. Default is 0.01.
            step_num (int): The number of steps to generate. Default is 100.

        Returns:
            Tensor: A tensor containing the beta values.
        """
        betas = torch.linspace(ddpm_start, ddpm_end, step_num)
        assert betas.numel() == step_num, "Number of generated betas does not match the step number."
        return betas

    def forward(self, data, t, pipeline, seed=None, **kwargs):
        """
        Apply the DDPM degradation process to the input data based on the degradation factor 't'.

        Args:
            pipeline (object): Pipeline object, expected to contain 'step_num' attribute.
            data (Tensor): The input tensor to be degraded.
            t (float or Tensor): The degradation factor. If it's a tensor, it should have the same batch size as the data.
                                 Each element of 't' should be in the range [0, 1].
                                 A value of 0 (resp. 1) means 0% (resp. 100%) degradation.
            seed (int, optional): A seed for random number generation to ensure reproducibility.

        Returns:
            Tensor: The degraded version of the input data.

        Raises:
            ValueError: If 't' has values outside the range [0, 1].
            TypeError: If 't' is not a float or a tensor.
        """
        assert pipeline is not None
        data_orig_shape = data.shape
        data = data.view(data.shape[0], -1)

        row_num = data.shape[0]
        feature_dim = data.shape[1]
        config = pipeline.config
        step_num = config.step_num

        # Validate and prepare 't'.
        if torch.is_tensor(t):
            t = t.reshape(row_num, 1)
            if not torch.all((t >= 0) & (t <= 1)):
                raise ValueError("All elements of tensor 't' must be between 0 and 1.")
        elif isinstance(t, float):
            if not (0 <= t <= 1):
                raise ValueError("The float value of 't' must be between 0 and 1.")
            t = torch.full((row_num, 1), t, device=data.device)
        else:
            raise TypeError("'t' must be a float or a tensor.")

        # Scale 't' according to the number of steps.
        t_scaled = torch.round(t * (step_num - 1.0)).long().to(data.device)
        betas = VectorDegradationDDPM.generate_schedule(step_num=step_num, ddpm_start=config.ddpm_start, ddpm_end=config.ddpm_end).to(data.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphabar_t = torch.gather(alphas_cumprod, 0, t_scaled.flatten()).view(row_num, 1)
        assert alphabar_t.numel() == row_num, "The number of elements in alphabar_t does not match row_num."

        # Compute mean and standard deviation for the noise.
        data_noise_mean = torch.sqrt(alphabar_t) * data  # element-wise multiplication
        data_noise_std = torch.sqrt(1.0 - alphabar_t)  # This is a column vector
        data_noise_std = data_noise_std.repeat(1, feature_dim)  # Repeat to match data dimensions

        # Generate noise and apply degradation.
        noise = rand_like_with_seed(data, seed=seed)
        data_noise = data_noise_mean + data_noise_std * noise

        data_noise = data_noise.view(data_orig_shape)
        return data_noise









class VectorDegradationIncreaseVariance:

    def __init__(self):
        pass

    def __call__(self, data, t, pipeline, seed=None, **kwargs):
        if torch.is_tensor(t):
            batch_dim = data.shape[0]
            t = t.reshape(batch_dim, 1)
            if not torch.all((t >= 0) & (t <= 1)):
                raise ValueError("All elements of tensor 't' must be between 0 and 1.")
        elif isinstance(t, float):
            if not (0 <= t <= 1):
                raise ValueError("The float value of 't' must be between 0 and 1.")
        else:
            raise TypeError("'t' must be a float or a tensor.")

        if isinstance(t, float) and t < 1e-7:
            # If 't' is very small, return the data unmodified as the degradation is negligible.
            return data


        data_orig_shape = data.shape
        data = data.view(data.shape[0], -1)

        std_dev = t*10
        # Generate a sample from the standard normal distribution.
        standard_normal_sample = rand_like_with_seed(data, seed=seed)

        # Apply the degradation transformation.
        transformed_sample = data + std_dev * standard_normal_sample

        transformed_sample = transformed_sample.view(data_orig_shape)
        return transformed_sample
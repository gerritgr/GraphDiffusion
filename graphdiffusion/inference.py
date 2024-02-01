import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())

import torch
import numpy as np


class VectorInference:
    """
    A class for performing vector inference, which typically involves reversing a degradation process.

    The class uses a specified 'pipeline' to manage the model, degradation, reconstruction, and bridging operations.
    The inference iteratively refines the estimate of the original data starting from noise.

    Attributes:
        pipeline (object): An object encapsulating the model and methods for degradation, reconstruction, and bridging.
    """

    def __init__(self):
        """
        Initializes the VectorInference class.

        Args:
            pipeline (object, optional): An object representing the processing pipeline.
        """

    def __call__(self, data, pipeline, noise_to_start, steps):
        """
        Perform vector inference by iteratively refining the estimate of the original data.

        Args:
            data (torch.utils.data.DataLoader, optional): A DataLoader providing data to infer from. Required if 'noise_to_start' is not provided.
            noise_to_start (Tensor, optional): The starting noise tensor to begin inference. Required if 'dataloader' is not provided.
            steps (int or array-like, optional): The number of steps or a sequence of 't' values for the inference process. Defaults to a linspace from 1 to 0 with 100 steps.

        Returns:
            tuple: A tuple containing the tensor of the last inference step and the reconstructed tensor from the last step.

        Raises:
            ValueError: If the steps are not in a decreasing sequence or not in the range [0, 1], or if neither 'dataloader' nor 'noise_to_start' is provided.
        """
        # Generate or validate the steps sequence
        assert data is None or noise_to_start is None, "Either data or noise_to_start must be provided, but not both."
        # assert isinstance(pipeline, PipelineEuclid), "pipeline must be a PipelineEuclid" # TODO add basepipeline class
        self.pipeline = pipeline

        if isinstance(steps, int):
            steps = np.linspace(1, 0, steps)
        elif steps is None:
            step_num = pipeline.config["step_num"] or 100
            steps = np.linspace(1, 0, step_num)

        # Validate the steps
        if len(steps) < 2 or not np.all(steps[:-1] > steps[1:]) or not (0 <= steps[0] <= 1 and 0 <= steps[-1] <= 1):
            raise ValueError("Steps must be a decreasing sequence in the range [0, 1] and have a length of at least 2.", "steps:", steps)

        # Ensure either dataloader or noise_to_start is provided
        if data is None and noise_to_start is None:
            raise ValueError("Either data or noise_to_start must be provided.")

        # Set the model to evaluation mode
        model = self.pipeline.get_model()
        model.eval()

        # Initialize the starting point by adding 100% noise to a random data point if noise_to_start is not provided
        if noise_to_start is None:
            if isinstance(data, torch.utils.data.DataLoader):
                random_data_point = next(iter(data))
            elif isinstance(data, torch.Tensor):
                random_data_point = data
            else:
                raise ValueError("data must be a DataLoader or a Tensor")
            noise_to_start = self.pipeline.degradation(random_data_point, t=1.0)

        data_t = noise_to_start  # Initialize data_t with the starting noise
        for i, t in enumerate(steps):
            data_0 = self.pipeline.reconstruction(data_t, t)  # Reconstruct data from the current state
            if i >= len(steps) - 1:  # Check if it's the last step
                if t < 1e-3:
                    return data_0, data_0
                else:
                    return data_t, data_0  # Return the last state and its reconstruction

            # Compute the next state
            t_next = steps[i + 1]
            assert t_next < t, "Next step should be smaller than the current step"
            data_t_next = self.pipeline.bridge(data_t, data_0, t, t_next)
            data_t = data_t_next

            if torch.isnan(data_t).any() or torch.isinf(data_t).any():
                nan_inf_mask = torch.isnan(data_t) | torch.isinf(data_t)
                count_nan_inf = torch.sum(nan_inf_mask).item()  # Count the number of NaN/Inf values
                warn_str = f"Warning at step {i}: The tensor contains {count_nan_inf} NaN/Inf values. These will be replaced with 0."
                warnings.warn(warn_str)
                data_t = torch.where(nan_inf_mask, torch.zeros_like(data_t), data_t)

        return data_t, data_0

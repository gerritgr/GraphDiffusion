import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())

import torch.nn.functional as F

import torch
import torch.nn.functional as F


class VectorDistance:
    """
    A class to compute distances between vectors.

    This class supports two types of distances:
    1. L1 (Manhattan) distance
    2. L2 (Euclidean) distance

    Although this class does not inherit from nn.Module like typical PyTorch modules,
    it is designed to function similarly by utilizing the __call__ method.

    Attributes:
        pipeline (object, optional): An optional attribute, not used in the current implementation.
                                    It's reserved for future use if there's a need to integrate this module
                                    within a specific processing pipeline.

    Methods:
        __call__(pipeline, x1, x2, dist_type="L1", *args, **kwargs):
            Computes the distance between two vectors.

    Note:
        The __call__ method is a special method in Python classes. It allows an instance of a class
        to be called as a function. In this case, it is used to compute the distance between two vectors.
    """

    def __init__(self):
        """
        Initializes the VectorDistance class.

        Args:
            pipeline (object, optional): Pipeline object for future use.
        """
        pass

    def __call__(self, x1, x2, dist_type="L1", *args, **kwargs):
        """
        Compute the distance between two vectors based on the specified distance type.

        Args:
            pipeline (object): Pipeline object, unused.
            x1 (Tensor): The first input tensor.
            x2 (Tensor): The second input tensor, must be the same size as x1.
            dist_type (str, optional): The type of distance to compute. Supported values are "L1" and "L2".
                                       Defaults to "L1".

        Returns:
            Tensor: A tensor containing the computed distance.

        Raises:
            ValueError: If the 'dist_type' is not recognized (not "L1" or "L2").
        """
        if dist_type == "L2":
            # Using the built-in mse_loss function for Euclidean distance (L2 norm)
            # The square root of MSE gives the Euclidean distance
            return torch.sqrt(F.mse_loss(x1, x2, reduction="mean"))
        elif dist_type == "L1":
            # Using the built-in l1_loss function for Manhattan distance (L1 norm)
            # It computes the mean absolute error (MAE) between x1 and x2
            return F.l1_loss(x1, x2, reduction="mean")
        else:
            # If the distance type is neither "L1" nor "L2", raise an error
            raise ValueError(f"Unsupported distance type: {dist_type}")

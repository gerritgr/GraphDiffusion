# import os
# os.chdir("../graphdiffusion")
# print("Current Working Directory: ", os.getcwd())

import os, sys

# os.chdir("")
sys.path.append("../graphdiffusion")

with open("../graphdiffusion/imports.py", "r") as file:
    exec(file.read())

from pipeline import *


############
# Spiral
############

import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_noisy_spiral(num_points=1000, noise_level=0.2, scale=2.0):
    from sklearn.datasets import make_swiss_roll

    # Generate a Swiss Roll dataset
    X, _ = make_swiss_roll(n_samples=num_points, noise=noise_level)  # Swiss Roll in 3D

    # Projection to 2D: we'll just use two of the three dimensions to create a 2D spiral
    X_2d = X[:, [0, 2]]

    # Normalize the data to center the spiral
    X_2d -= X_2d.mean(axis=0)

    # Convert to torch.Tensor
    X_2d_noisy = torch.tensor(X_2d, dtype=torch.float32)

    # Normalize the data to be between -1 and 1
    min_val, _ = torch.min(X_2d_noisy, dim=0, keepdim=True)
    max_val, _ = torch.max(X_2d_noisy, dim=0, keepdim=True)
    X_2d_normalized = 2 * ((X_2d_noisy - min_val) / (max_val - min_val)) - 1

    # Convert to a list of 2-element tensors
    noisy_spiral_data_list = [tensor * scale for tensor in X_2d_normalized]

    return noisy_spiral_data_list


# Generate the points
points = generate_noisy_spiral()

# Plotting
plt.clf()
plt.scatter([p[0].item() for p in points], [p[1].item() for p in points], alpha=0.6, s=50)
plt.title("Noisy 2D Spiral")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")


plt.savefig("spiral.png")


############
# Inspect Data
############


def plot_2darray_on_axis(array, axis, arrays):
    """
    Draws the input 2D NumPy array on the provided Matplotlib axis using a scatter plot.
    Each row in the 2D array is considered a separate point in the scatter plot.
    If the input array is 1D, it's treated as a single point with batch size 1.

    :param array: A 2D or 1D NumPy array of numbers to be plotted.
                  If 2D, shape should be (batch_size, 2).
                  If 1D, shape should be (2,).
    :param axis: A Matplotlib axis object on which to draw the scatter plot.
    :param arrays: List of all arrays that will be plotttet.
    """
    # If the input array is 1D, reshape it to 2D with batch size 1
    if array.ndim == 1:
        array = array.reshape(1, -1)

    # Ensure array is a 2D tensor after reshaping
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Input array must be a 2D or 1D array with shape (batch_size, 2) or (2,)")

    if np.isnan(array).any() or np.isinf(array).any():
        warnings.warn("Input array contains NaN or Inf values. These will be replaced with 0.")
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    # Plotting the scatter plot on the provided axis for all points
    axis.scatter(array[:, 0], array[:, 1], s=100, alpha=0.5, edgecolors="none")

    # Set x and y axis limits
    min_x = np.min([np.min(x[:, 0]) for x in arrays])
    max_x = np.max([np.max(x[:, 0]) for x in arrays])
    min_y = np.min([np.min(x[:, 1]) for x in arrays])
    max_y = np.max([np.max(x[:, 1]) for x in arrays])
    # axis.set_xlim([min_x,max_x])
    # axis.set_ylim([min_y,max_y])

    # Setting labels
    axis.set_xlabel("x")
    axis.set_ylabel("y")


train_dataloader = DataLoader(points, batch_size=100, shuffle=True)
pipeline = VectorPipeline(node_feature_dim=2)
pipeline.visualize_foward(
    data=train_dataloader,
    outfile="spiral_forward.jpg",
    plot_data_func=plot_2darray_on_axis,
    num=25,
)


############
# Train
############

train_dataloader = DataLoader(points, batch_size=100, shuffle=True)
print("first ten points", points[:10])
#pipeline.train(train_dataloader, epochs=10000)
#pipeline.reconstruction_obj.save_model(pipeline=pipeline, pre_trained_path="../pre_trained/vectordenoiser_spiral_weights.pt")

############
# Inference
############
print("start inference")
train_dataloader = DataLoader(points, batch_size=100, shuffle=True)
pipeline = VectorPipeline(pre_trained="../pre_trained/vectordenoiser_spiral_weights.pt", node_feature_dim=2)
data0 = pipeline.inference(data=train_dataloader)
pipeline.visualize_reconstruction(
    data=train_dataloader,
    plot_data_func=plot_2darray_on_axis,
    outfile="spiral_backward.jpg",
    num=25,
    steps=100,
)


compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="spiral_compare.pdf")
print("compare linear", compare)

############
# Denoising Diffusion Forward Process
############

# Forward
degradation_obj = VectorDegradationDDPM()
bridge_obj = VectorBridgeDDPM()
# pipeline = VectorPipeline(pre_trained="../pre_trained/vectordenoiser_spiral_weights_ddpm.pt", node_feature_dim=2, degradation_obj=degradation_obj, bridge_obj=bridge_obj)
pipeline = VectorPipeline(node_feature_dim=2, degradation_obj=degradation_obj, bridge_obj=bridge_obj)
pipeline.visualize_foward(
    data=train_dataloader,
    outfile="spiral_forward_ddpm.jpg",
    plot_data_func=plot_2darray_on_axis,
    num=25,
)

# Train
# pipeline.train(train_dataloader, epochs=10000)
# pipeline.reconstruction_obj.save_model(pipeline=pipeline, pre_trained_path="../pre_trained/vectordenoiser_spiral_weights_ddpm.pt")
pipeline = VectorPipeline(
    pre_trained="../pre_trained/vectordenoiser_spiral_weights_ddpm.pt",
    node_feature_dim=2,
    degradation_obj=degradation_obj,
    bridge_obj=bridge_obj,
)

# Inference
pipeline.visualize_reconstruction(
    data=train_dataloader,
    plot_data_func=plot_2darray_on_axis,
    outfile="spiral_backward_ddpm.jpg",
    num=25,
    steps=100,
)


compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="spiral_compare_ddpm.pdf")
print("compare ddpm", compare)
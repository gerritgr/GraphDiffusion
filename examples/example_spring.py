#import os
#os.chdir("../graphdiffusion")
#print("Current Working Directory: ", os.getcwd())

import os, sys
#os.chdir("")
sys.path.append("../graphdiffusion")

with open('../graphdiffusion/imports.py', 'r') as file:
    exec(file.read())
      
from pipeline import *



############
# Spiral
############

import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_noisy_spiral(num_points=1000, noise_level=5.5):
    """
    Generate points distributed according to a noisy 2D spiral.

    :param num_points: The number of data points to generate.
    :param noise_level: The standard deviation of the noise.
    :return: A list of 2D PyTorch tensors, each representing a point (x, y).
    """
    # Generate a linearly increasing vector of angles
    theta = np.linspace(0, 4 * np.pi, num_points)  # 4*pi for two complete rotations

    # The radius of the spiral increases as a function of theta
    r = theta**2

    # Convert from polar to Cartesian coordinates
    x = r * np.cos(theta) + np.random.normal(0, noise_level, num_points)
    y = r * np.sin(theta) + np.random.normal(0, noise_level, num_points)

    # Combine x and y into a list of 2D PyTorch tensors
    points = [torch.tensor([x[i], y[i]], dtype=torch.float32) for i in range(num_points)]

    return points

# Generate the points
points = generate_noisy_spiral()

# Plotting
plt.clf()
plt.scatter([p[0].item() for p in points], [p[1].item() for p in points], alpha=0.6, s=500)
plt.title('Noisy 2D Spiral')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')


plt.savefig("spiral.png")

############
# Inspect Data
############

def plot_2darray_on_axis(array, axis, x_limits, y_limits):
    """
    Draws the input 2D NumPy array on the provided Matplotlib axis using a scatter plot.
    Each row in the 2D array is considered a separate point in the scatter plot.
    If the input array is 1D, it's treated as a single point with batch size 1.

    :param array: A 2D or 1D NumPy array of numbers to be plotted.
                  If 2D, shape should be (batch_size, 2).
                  If 1D, shape should be (2,).
    :param axis: A Matplotlib axis object on which to draw the scatter plot.
    :param x_limits: A tuple containing the minimum and maximum x-axis limits.
    :param y_limits: A tuple containing the minimum and maximum y-axis limits.
    """
    # If the input array is 1D, reshape it to 2D with batch size 1
    if array.ndim == 1:
        array = array.reshape(1, -1)

    # Ensure array is a 2D tensor after reshaping
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError('Input array must be a 2D or 1D array with shape (batch_size, 2) or (2,)')

    # Plotting the scatter plot on the provided axis for all points
    axis.scatter(array[:, 0], array[:, 1], s=1000, alpha=0.5, edgecolors='none')

    # Set x and y axis limits
    #axis.set_xlim(x_limits)
    #axis.set_ylim(y_limits)

    # Setting labels
    axis.set_xlabel('x')
    axis.set_ylabel('y')


train_dataloader = DataLoader(points, batch_size=50, shuffle=True)
pipeline = VectorPipeline(node_feature_dim=2)
pipeline.visualize_foward(train_dataloader, outfile="spiral_forward.jpg", plot_data_func=plot_2darray_on_axis, num=25)


############
# Train
############



print("first ten points", points[:10])
pipeline.train(train_dataloader, epochs=100000)
pipeline.reconstruction_obj.save_model(pipeline, "../pre_trained/vectordenoiser_spiral_weights.pt")

############
# Inference
############

pipeline = VectorPipeline(pre_trained="../pre_trained/vectordenoiser_spiral_weights.pt", node_feature_dim=2)
data0 = pipeline.inference(train_dataloader)
print("inference spring", data0)

pipeline.visualize_reconstruction(data=train_dataloader, plot_data_func=plot_2darray_on_axis, outfile="spiral_backward.jpg")

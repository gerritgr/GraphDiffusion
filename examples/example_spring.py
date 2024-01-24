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

def generate_noisy_spiral(num_points=10, noise_level=5.5):
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
# Train
############

pipeline = VectorPipeline(node_feature_dim=2)
#tensorlist = TensorDataset(*points)
data = DataLoader(points, batch_size=1, shuffle=True)
print(points[:10])
pipeline.train(data, epochs=100)
pipeline.reconstruction_obj.save_model(pipeline, "../pre_trained/vectordenoiser_spiral_weights.pt")

############
# Inference
############

pipeline = VectorPipeline(pre_trained="../pre_trained/vectordenoiser_spiral_weights.pt", node_feature_dim=2)
data0 = pipeline.inference(data)
print("inference spring", data0)



def plot_2darray_on_axis(array, axis, x_limits, y_limits):
    """
    Draws the input 1D PyTorch array on the provided Matplotlib axis using a scatter plot.

    :param array: A 1D PyTorch array of numbers to be plotted.
    :param axis: A Matplotlib axis object on which to draw the scatter plot.
    :param x_limits: A tuple containing the minimum and maximum x-axis limits.
    :param y_limits: A tuple containing the minimum and maximum y-axis limits.
    """
    array = array.detach().numpy()

    axis.scatter([array[0]], [array[1]], s=1000, alpha=0.5, edgecolors='none')   # Plotting the scatter plot on the provided axis
    axis.set_xlim([-10,10])  # Set x-axis limits
    axis.set_ylim([-10,10])  # Set x-axis limits
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    #axis.set_title('Scatter Plot')

pipeline.visualize_foward(points[0], outfile="spiral_forward.jpg", plot_data_func=plot_2darray_on_axis, num=25)
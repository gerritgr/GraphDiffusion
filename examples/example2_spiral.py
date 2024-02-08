import os, sys
from pathlib import Path

with open("../graphdiffusion/imports.py", "r") as file:
    exec(file.read())
# Add the parent directory of the script to sys.path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent  # this should point to 'graphdiffusion/'
sys.path.insert(0, str(parent_dir))


from graphdiffusion.pipeline import *
from graphdiffusion.bridge import *
from graphdiffusion.degradation import *
from graphdiffusion.reconstruction import *
from graphdiffusion.distance import *

COMPARE = True


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
plt.savefig("images/example2_spiral.png")



############
# Inspect Data
############


train_dataloader = DataLoader(points, batch_size=100, shuffle=True)
pipeline = PipelineVector(node_feature_dim=2, dist_type="L2", pre_trained_path="../pre_trained/vectordenoiser_spiral_weights.pt", level="DEBUG")
pipeline.visualize_foward(
    data=train_dataloader,
    outfile="images/example2_spiral_forward.jpg",
    num=25,
)


############
# Train
############

train_dataloader = DataLoader(points, batch_size=500, shuffle=True)
pipeline.train(data=train_dataloader, epochs=100000)
pipeline.save_all_model_weights("../pre_trained/vectordenoiser_spiral_weights.pt")

############
# Inference
############
compare_dataloader = DataLoader(points, batch_size=100, shuffle=True)
#pipeline = PipelineVector(node_feature_dim=2, pre_trained_path="../pre_trained/vectordenoiser_spiral_weights.pt")


pipeline.visualize_reconstruction(
    data=compare_dataloader,
    outfile="images/example2_spiral_backward.jpg",
    num=25,
    steps=100,
)

if COMPARE:
    compare = pipeline.compare_distribution(real_data=compare_dataloader, outfile="images/example2_spiral_compare.pdf")
    print("compare linear", compare)


############
# Denoising Diffusion Forward Process
############

# Forward
degradation_obj = VectorDegradationDDPM()
bridge_obj = VectorBridgeDDPM()
pipeline = PipelineVector(pre_trained_path="../pre_trained/vectordenoiser_spiral_weights_ddpm.pt", node_feature_dim=2, degradation_obj=degradation_obj, bridge_obj=bridge_obj)
#pipeline = PipelineVector(node_feature_dim=2, degradation_obj=degradation_obj, bridge_obj=bridge_obj)
pipeline.visualize_foward(
    data=train_dataloader,
    outfile="images/example2_spiral_forward_ddpm.jpg",
    #    plot_data_func=plot_2darray_on_axis,
    num=25,
)

# Train
pipeline.train(data=train_dataloader, epochs=100000)
pipeline.save_all_model_weights("../pre_trained/vectordenoiser_spiral_weights_ddpm.pt")

pipeline = PipelineVector(
    pre_trained_path="../pre_trained/vectordenoiser_spiral_weights_ddpm.pt",
    node_feature_dim=2,
    degradation_obj=degradation_obj,
    bridge_obj=bridge_obj,
)

# Inference
pipeline.visualize_reconstruction(
    data=train_dataloader,
    plot_data_func=plot_2darray_on_axis,
    outfile="images/example2_spiral_backward_ddpm.jpg",
    num=25,
    steps=100,
)

if COMPARE:
    compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="images/example2_spiral_compare_ddpm.pdf")
    print("compare ddpm", compare)
    compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="images/example2_spiral_compare_ddpm_emd.pdf", method="emd")
    print("compare ddpm emd", compare)


if False:
    ############
    # High Variance Process
    ############

    train_dataloader = DataLoader(points, batch_size=1000, shuffle=True)
    degradation_obj = VectorDegradationHighVariance(std_dev_max=1.5, time_scaling_factor=2.0)
    pipeline = PipelineVector(node_feature_dim=2, degradation_obj=degradation_obj)
    pipeline.visualize_foward(
        data=train_dataloader,
        outfile="images/example2_spiral_forward_hv.jpg",
        num=25,
    )
    pipeline.config["vectorbridge_magnitude_scale"] = 0.9
    pipeline.config["vectorbridge_rand_scale"] = 5.0

    train_dataloader = DataLoader(points[100:], batch_size=100, shuffle=True)
    test_dataloader = DataLoader(points[:100], batch_size=100, shuffle=True)

    pipeline.train(data=train_dataloader, epochs=100000, data_test=test_dataloader)
    pipeline.save_all_model_weights("../pre_trained/vectordenoiser_spiral_weights_hv.pt")

    pipeline.visualize_reconstruction(
        data=train_dataloader,
        plot_data_func=plot_2darray_on_axis,
        outfile="images/example2_spiral_backward_hv.jpg",
        num=25,
        steps=100,
    )

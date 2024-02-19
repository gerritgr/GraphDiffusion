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
from graphdiffusion.utils import *

set_all_seeds(1234)

COMPARE = True


# check this https://github.com/gerritgr/GraphDiffusion/blob/e3fffe68f22b64a5204c2d575f51fe3ca11bc400/examples/example_spring.py

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
plt.savefig(create_path("images/example2/spiral.png"))


############
# Inspect Data
############


train_dataloader = DataLoader(points, batch_size=100, shuffle=True)
pipeline = PipelineVector(node_feature_dim=2, pre_trained_path="../pre_trained/vectordenoiser_spiral_weights.pt", level="DEBUG")
pipeline.visualize_foward(
    data=train_dataloader,
    outfile="images/example2/spiral_forward.jpg",
    num=25,
)


############
# Train
############

train_dataloader = DataLoader(points, batch_size=100, shuffle=True)
# pipeline.train(data=train_dataloader, epochs=100000)
# pipeline.save_all_model_weights("../pre_trained/vectordenoiser_spiral_weights.pt")

############
# Inference
############


pipeline.visualize_reconstruction(
    data=train_dataloader,
    outfile="images/example2/spiral_backward.jpg",
    num=25,
    steps=100,
)

if COMPARE:
    compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="images/example2/spiral_compare_linear.pdf")
    pipeline.info("compare linear", compare)
    compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="images/example2/spiral_compare_linear_emd.pdf", method="emd")
    pipeline.info("compare linear emd", compare)


############
# Different Inference Methods
############

pipeline.visualize_reconstruction(
    data=train_dataloader,
    outfile="images/example2/spiral_backward_long.jpg",
    num=25,
    steps=1000,
)


pipeline.bridge_obj = VectorBridgeBackup()
pipeline.visualize_reconstruction(
    data=train_dataloader,
    outfile="images/example2/spiral_backward_backup.jpg",
    num=25,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeNaive()
pipeline.visualize_reconstruction(
    data=train_dataloader,
    outfile="images/example2/spiral_backward_naive.jpg",
    num=25,
    steps=100,
)

pipeline.bridge_obj = VectorBridgeColdDiffusion()
pipeline.visualize_reconstruction(
    data=train_dataloader,
    outfile="images/example2/spiral_backward_cold.jpg",
    num=25,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeAlt()
pipeline.visualize_reconstruction(
    data=train_dataloader,
    outfile="images/example2/spiral_backward_alt.jpg",
    num=25,
    steps=100,
)


####################################
####################################
# Denoising Diffusion Forward Process
####################################
####################################


# Forward
degradation_obj = VectorDegradationDDPM()
bridge_obj = VectorBridgeDDPM()
pipeline = PipelineVector(pre_trained_path="../pre_trained/vectordenoiser_spiral_weights_ddpm.pt", node_feature_dim=2, degradation_obj=degradation_obj, bridge_obj=bridge_obj)
pipeline.visualize_foward(
    data=train_dataloader,
    outfile="images/example2/spiral_forward_ddpm.jpg",
    #    plot_data_func=plot_2darray_on_axis,
    num=25,
)

# Train
# pipeline.train(data=train_dataloader, epochs=100000)
# pipeline.save_all_model_weights("../pre_trained/vectordenoiser_spiral_weights_ddpm.pt")

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
    outfile="images/example2/spiral_backward_ddpm.jpg",
    num=25,
    steps=100,
)

if COMPARE:
    compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="images/example2/spiral_compare_ddpm.pdf")
    pipeline.info("compare ddpm", compare)
    compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="images/example2/spiral_compare_ddpm_emd.pdf", method="emd")
    pipeline.info("compare ddpm emd", compare)



####################################
####################################
# HV Method
####################################
####################################


train_dataloader = DataLoader(points, batch_size=100, shuffle=True)
degradation_obj = VectorDegradationIncreaseVariance()
pipeline = PipelineVector(node_feature_dim=2, degradation_obj=degradation_obj, pre_trained_path="../pre_trained/vectordenoiser_spiral_weights_iv.pt")
pipeline.bridge_obj = VectorBridgeAlt()
pipeline.visualize_foward(
    data=train_dataloader,
    outfile="images/example2/spiral_forward_iv.jpg",
    num=25,
)

pipeline.visualize_foward(
    data=train_dataloader,
    outfile="images/example2/spiral_forward_iv_normal.jpg",
    num=25,
    plot_data_func=plot_data_as_normal_pdf,
)

train_dataloader = DataLoader(points[10:], batch_size=100, shuffle=True)
test_dataloader = DataLoader(points[:10], batch_size=100, shuffle=True)

pipeline.train(data=train_dataloader, epochs=10000, data_test=test_dataloader)
pipeline.save_all_model_weights("../pre_trained/vectordenoiser_spiral_weights_iv.pt")

pipeline.visualize_reconstruction(
    data=train_dataloader,
    plot_data_func=plot_2darray_on_axis,
    outfile="images/example2/spiral_backward_iv.jpg",
    num=25,
    steps=100,
)


compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="images/example2/spiral_compare_iv.pdf")
pipeline.info("compare iv", compare)


#VectorDenoiserLarge


pipeline = PipelineVector(node_feature_dim=2, degradation_obj=degradation_obj, reconstruction_obj = VectorDenoiserLarge(node_feature_dim=2), pre_trained_path="../pre_trained/vectordenoiser_spiral_weights_iv2.pt", level="DEBUG")
pipeline.bridge_obj = VectorBridgeAlt()


pipeline.train(data=train_dataloader, epochs=10000, data_test=test_dataloader)
pipeline.save_all_model_weights("../pre_trained/vectordenoiser_spiral_weights_iv2.pt")

pipeline.visualize_reconstruction(
    data=train_dataloader,
    plot_data_func=plot_2darray_on_axis,
    outfile="images/example2/spiral_backward_iv2.jpg",
    num=25,
    steps=100,
)


compare = pipeline.compare_distribution(real_data=train_dataloader, outfile="images/example2/spiral_compare_iv2.pdf")
pipeline.info("compare iv2", compare)
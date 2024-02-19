from graphdiffusion.pipeline import *
import seaborn as sns
from graphdiffusion.plotting import plot_data_as_normal_pdf
from graphdiffusion import utils

with open("../graphdiffusion/imports.py", "r") as file:
    exec(file.read())
import os

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, IterableDataset, TensorDataset
from PIL import Image

IMG_SIZE = 16


pipeline = PipelineImage(img_height=IMG_SIZE, img_width=IMG_SIZE, channels=3, time_scaling_factor=0.8, clamp_inference=(-1.1, 1.1), pre_trained_path="../pre_trained/mnist_weights.pt")
pipeline.visualize_foward(
    data=(torch.rand(1, 1, IMG_SIZE, IMG_SIZE) - 0.5) * 2.0,
    outfile="images/example6/noise_forward.jpg",
    plot_data_func=plot_image_on_axis,
    num=25,
)


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


simple_transform = get_simple_image_transform(convert_to_grayscale=False, width=16, height=16, convert_to_rgb=True)

from torch.utils.data import DataLoader, Subset
# Load the FashionMNIST dataset
mnist_train = datasets.MNIST(
    root="./data",  # Specify the path to store the data
    train=True,  # Load the training data
    download=True,  # Download the data if it's not already downloaded
    transform=simple_transform,  # Apply the defined transformations
)


num_samples = 1000
indices = torch.randperm(len(mnist_train)).tolist()[:num_samples]
subset_mnist_train = Subset(mnist_train, indices)

# Create a DataLoader
train_loader = DataLoader(
    subset_mnist_train,  # Dataset to load
    batch_size=100,  # Batch size
    shuffle=True,  # Shuffle the dataset
)



# Example: Iterate over the DataLoader
for images, labels in train_loader:
    print("Shape of image: ", images.shape)  # Should print [100, 3, 28, 28] indicating batch_size, channels, height, width
    break  # Break after the first batch to demonstrate


pipeline.visualize_foward(
    data=train_loader,
    outfile="images/example6/fmnist_forward.jpg",
    plot_data_func=plot_image_on_axis,
    num=25,
)


pipeline.visualize_foward(
    data=train_loader,
    outfile="images/example6/fmnist_forward2.jpg",
    plot_data_func=plot_data_as_normal_pdf,
    num=25,
)


pipeline.train(data=train_loader, epochs=10)
pipeline.save_all_model_weights("../pre_trained/mnist_weights.pt")





dataloader_show = DataLoader(
    subset_mnist_train,  # Dataset to load
    batch_size=1,  # Batch size
    shuffle=True,  # Shuffle the dataset
)


pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example6/backward_ddpm.jpg",
    num=36,
    steps=100,
)



pipeline.bridge_obj = VectorBridge()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example6/backward_normal.jpg",
    num=36,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeNaive()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example6/backward_naive.jpg",
    num=36,
    steps=100,
)


pipeline.bridge_obj = VectorBridgeColdDiffusion()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example6/backward_cold.jpg",
    num=36,
    steps=100,
)



pipeline.bridge_obj = VectorBridgeAlt()
pipeline.visualize_reconstruction(
    data=dataloader_show,
    plot_data_func=plot_image_on_axis,
    outfile="images/example6/backward_alt.jpg",
    num=36,
    steps=100,
)
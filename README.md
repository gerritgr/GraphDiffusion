# Do no use - GraphDiffusion
[![codecov](https://codecov.io/gh/gerritgr/GraphDiffusion/graph/badge.svg?token=O1FXPKS2ZI)](https://codecov.io/gh/gerritgr/GraphDiffusion)
![example workflow](https://github.com/github/docs/actions/workflows/multi_test.yml/badge.svg)

**Note**: This project is in the early stages of development and is currently public for development purposes only. It is not yet ready for production use.

GraphDiffusion is a repository for the `graphdiffusion`, a Python/PyTorch package designed for educational purposes. It focuses on the application of diffusion models to graphs and point clouds, offering a novel approach to understanding and manipulating these data structures.

## Features

- **PyTorch Integration**: Built on PyTorch and PyTorch Geometric, leveraging their powerful and flexible computing frameworks.
- **User-Friendly**: Extremely easy to use, offering off-the-shelf commands for a variety of use cases.
- **Versatile Data Handling**: Compatible with various data types, including:
    - Vectors (normal PyTorch Tensors)
    - Images
    - Graphs
    - Point Clouds
    - Molecules
- **Wasserstein Distance**: Employs the Wasserstein distance metric for automatic comparison between the distributions of original and generated data.
- **Rich Visualization**: Offers diverse options for visualizing both the degradation/forward process and the reconstruction/reverse process.
- **Extensibility**: Designed with flexibility in mind, allowing users to easily integrate their own Python functions into the framework.

We support multiple types of flow models, including:
- denoising diffusion probabilistic models (DDPM)
- normalizing flow models (NFM)
- conditional flow
- SDE-based models



## Design Choices

- **Degradation and Reconstruction**: This approach is favored over the typical forward and backward process, allowing a more intuitive understanding of the diffusion process.
- **Time Scale**: The process is measured from 0 (original sample) to 1 (completely destructed, e.g., random noise), offering a clear, linear progression.
- **Degradation/Forward Process**: This does not need to be a stochastic (Markov) process. Instead, you can jump directly to a randomly sampled time `t`.
- **Scheduler**: The model does not require a separate scheduler. The degradation process implicitly handles scheduling.

### Components

## How to Use

### Install Locally from GitHub

1. Install [Anaconda](https://www.anaconda.com/products/individual).
2. Create and activate a new environment, then install the package. Ensure you are in the topmost 'graphdiffusion' directory:

    ```bash
    conda env create -f environment.yml -n graphdiffusionenv
    conda activate graphdiffusionenv 
    pip install -e .
    ```

3. To run the example code, navigate to the examples directory:

    ```bash
    cd examples
    python example.py
    ```

   To run the tests, execute the following command in the topmost 'graphdiffusion' directory:

    ```bash
    python -m pytest
    ```

### Install Package

Currently not supported.

### Via Docker and Jupyter Lab

1. Install [Docker](https://docs.docker.com/get-docker/).
2. Run the following command:

    ```bash
    docker run -p 8888:8888 gerritgr/graphdiffusion:latest
    ```

   After running the Docker image, manually copy the provided URL to your browser. Navigate to the example folder and activate the _graphdiffusion_env_ kernel (`Kernel -> Change Kernel...` in Jupyter Lab).

### Via Colab
Not implemented yet. 


## Getting Started

(Provide a brief introduction to the project, its goals, and how users can quickly get started with it.)

### Tutorials

1. Spiral example.
2. Compontents of the diffusion pipeline.
2. Image example.
3. Graph example.
4. Molecule example.
4. Point cloud example.
5. Conditional models.
6. Hyperparameter tuning and WandB integration.



### Documentation

todo




### Roadmap

- ImagePipeline
- add time dim and conditioning dim to the pipeline
- add "create from data" method to the pipeline
- graphs
- wandb
- conditioning
- better config
- ability to save/pickle and load the whole pipeline
- include readthedocs



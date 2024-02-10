import os
import sys
import random
import torch
import pytest
from pathlib import Path

# current_dir = Path(__file__).parent
# parent_dir = current_dir.parent  # this should point to 'graphdiffusion/'
# sys.path.insert(0, str(parent_dir))

from graphdiffusion.pipeline import PipelineVector


@pytest.fixture
def pipeline():
    # Initialize your PipelineVector here with the desired parameters
    return PipelineVector(node_feature_dim=5)


def test_train(pipeline):
    example_tensor = torch.randn(5) * 10  # Replace with your actual data
    # You may need to mock the train method if it requires specific structure of data or parameters
    pipeline.train(example_tensor)
    # Include assertions here to validate the training process
    # For example: assert result is not None, if your train method returns a result


def test_reconstruction(pipeline):
    example_tensor = torch.randn(5) * 10
    x = pipeline.reconstruction(example_tensor, 0)
    # Include assertions here to validate the reconstruction
    assert x is not None


def test_bridge(pipeline):
    data_prediction = torch.randn(5) * 10
    x = pipeline.bridge(data_now=data_prediction * 0.5, data_prediction=data_prediction, t_now=1.0, t_query=0.5)
    # Include assertions here to validate the bridge functionality
    assert x is not None


def test_distance(pipeline):
    x1 = torch.tensor([5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01])
    x2 = torch.tensor([5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01])
    distance = pipeline.distance(x1, x2)
    # Include assertions here to validate the distance functionality
    assert distance is not None


def test_inference(pipeline):
    print("test inference")
    example_tensor = torch.randn(5, 5) * 10
    data0 = pipeline.inference(data=example_tensor, noise_to_start=None)
    # Include assertions here to validate the inference
    assert data0 is not None

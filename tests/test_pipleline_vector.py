from graphdiffusion.pipeline import *

import pytest
from torch import nn


def test_vector():
    pipeline = PipelineVector(node_feature_dim=5)
    example_tensor = torch.randn(5) * 10
    pipeline.train(example_tensor)
    x = pipeline.reconstruction(example_tensor, 0)
    x = pipeline.bridge(data_now=example_tensor * 0.5, data_prediction=example_tensor, t_now=1.0, t_query=0.5)

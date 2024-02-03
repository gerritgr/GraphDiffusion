from graphdiffusion.pipeline import *

import pytest 
from torch import nn 

def dummy_callable(x, x1=1, x2=2, x3=3, x4=4): 
    return x

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
    def forward(self, x, x1=1, x2=2, x3=3, x4=4):
        return None

def test_create_base():
    

    node_feature_dim = 2
    device = torch.device("cpu")
    reconstruction_obj = DummyModule()
    inference_obj = dummy_callable
    degradation_obj = dummy_callable
    train_obj = dummy_callable
    bridge_obj = dummy_callable
    distance_obj = dummy_callable
    encoding_obj = dummy_callable
    trainable_objects = None
    pre_trained_path = None
    
    pipeline = PipelineBase(node_feature_dim, device, reconstruction_obj, inference_obj, degradation_obj, train_obj, bridge_obj, distance_obj, encoding_obj, trainable_objects, pre_trained_path)
    return pipeline


def test_train_base():
    pipeline = test_create_base()
    example_tensor = torch.randn(5) * 10 
    pipeline.train(example_tensor)

def test_getmodel_base():
    pipeline = test_create_base()
    pipeline.get_model()
    with pytest.raises(AssertionError):
        pipeline.define_trainable_objects(True, True, True, True)

def test_getmodel_base():
    pipeline = test_create_base() 
    pipeline.get_model()
    pipeline.degradation_obj = DummyModule()
    pipeline.distance_obj = DummyModule()
    pipeline.encoding_obj = DummyModule()
    pipeline.define_trainable_objects(True, True, True, True)
    pipeline.get_model()


def test_default_methods_base():
    pipeline = test_create_base()
    x = pipeline.distance(1,1)
    x = pipeline.bridge(1,2,3,4)
    x = pipeline.inference(1,2)
    x = pipeline.reconstruction(1,2)
    x = pipeline.degradation(1,2)


from graphdiffusion.plotting import plot_data_as_normal_pdf

def test_forward_base():
    pipeline = test_create_base()
    input_tensor = torch.rand(5,5)
    pipeline.visualize_foward(input_tensor, outfile=None, num=9, plot_data_func=plot_data_as_normal_pdf)


def test_vector():
    pipeline = PipelineVector(node_feature_dim=5)
    example_tensor = torch.randn(5) * 10 
    pipeline.train(example_tensor)



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



# Example usage
pipeline = PipelineVector(node_feature_dim=5)

# Example tensor
example_tensor = torch.randn(5) * 10  # Replace with your actual data

# Using the train method
pipeline.train(example_tensor)

# Using the reconstruction method
x = pipeline.reconstruction(example_tensor, 0)
print("reconstruction", x)

x = pipeline.bridge(data_now=0.5, data_prediction=example_tensor, t_now=1.0, t_query=0.5)
print("bridge", x)

input_tensor = torch.rand(5,5)
pipeline.visualize_foward(input_tensor, outfile="images/example1_forward_5d.jpg")


x1 = [5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01]
x2 = [5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01]
print("loss", pipeline.distance(torch.tensor(x1), torch.tensor(x2)))

x3 = [5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01]
x4 = [5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01]


example_tensor = torch.randn(5,5) * 10
data0 = pipeline.inference(data=example_tensor, noise_to_start=None)
print("inference", data0)

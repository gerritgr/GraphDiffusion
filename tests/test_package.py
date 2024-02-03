from graphdiffusion.pipeline import *

# Example usage
pipeline = PipelineVector(node_feature_dim=5)

# Example tensor
example_tensor = torch.randn(5) * 10  # Replace with your actual data

# Using the train method
pipeline.train(example_tensor)
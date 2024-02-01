with open("imports.py", "r") as file:
    exec(file.read())

from pipeline import *

# Example usage
pipeline = PipelineEuclid(node_feature_dim=5)

# Example tensor
example_tensor = torch.randn(5) * 10  # Replace with your actual data

# Using the train method
# pipeline.train(example_tensor)

# Using the reconstruction method
x = pipeline.reconstruction(example_tensor, 0)
print("reconstruction", x)

x = pipeline.bridge(data_now=None, data_prediction=example_tensor, t_now=1.0, t_query=0.5)
print("bridge", x)

input_tensor = [5.0 + random.random() * 0.01, -5.0 + random.random() * 0.01]
random.shuffle(input_tensor)
input_tensor = torch.tensor(input_tensor)
print("input_tensor", input_tensor)
pipeline.visualize_foward(input_tensor)


x1 = [5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01]
x2 = [5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01]
print("loss", pipeline.distance(torch.tensor(x1), torch.tensor(x2)))

x3 = [5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01]
x4 = [5.0 + random.random() * 0.01, -5.0 - random.random() * 0.01]

pipeline = PipelineEuclid(node_feature_dim=2)
tensorlist = [torch.tensor(x1), torch.tensor(x2), torch.tensor(x3), torch.tensor(x4)]
# data = TensorDataset(*tensorlist)
data = DataLoader(tensorlist, batch_size=1, shuffle=True)
pipeline.train(data, epochs=10000)


data0 = pipeline.inference(data)
print("inference", data0)

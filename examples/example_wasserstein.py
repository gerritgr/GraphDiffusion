import numpy as np

import torch
import numpy as np
import ot  # Python Optimal Transport library
import matplotlib.pyplot as plt


# Define your custom compute_distance function
def compute_distance(x, y):
    # Implement your distance computation here
    # For example, Euclidean distance could be:
    return np.sqrt(np.sum((x - y) ** 2))



# Set random seed for reproducibility
np.random.seed(42)

# Define batch size
batch_size = 50

# Create datasets data_0 and data_1 using 2D standard normal distribution
data_0 = np.random.randn(batch_size, 2)
data_1 = np.random.randn(batch_size, 2)



# Convert PyTorch tensors to NumPy arrays if they are not already
if isinstance(data_0, torch.Tensor):
    data_0 = data_0.detach().cpu().numpy()
if isinstance(data_1, torch.Tensor):
    data_1 = data_1.detach().cpu().numpy()

# 1) Compute the cost matrix between the two datasets
# Using Euclidean distance here, but you can choose an appropriate measure
# cost_matrix = ot.dist(data_0, data_1, metric='euclidean')
cost_matrix = np.zeros((len(data_0), len(data_1)))

# Fill in the cost matrix using your compute_distance function
for i in range(len(data_0)):
    for j in range(len(data_1)):
        cost_matrix[i, j] = compute_distance(data_0[i], data_1[j])


# Compute the optimal transport plan (coupling matrix) using the Sinkhorn algorithm
# epsilon is the regularization term, you may need to adjust it
epsilon = 1e-2
coupling_matrix = ot.sinkhorn(a=np.ones(len(data_0)) / len(data_0), b=np.ones(len(data_1)) / len(data_1), M=cost_matrix, reg=epsilon)

# Find the indices of the mappings (this works for small enough epsilon and assumes that data_0 and data_1 have the same length)
# For each point in data_0, find the index of its corresponding point in data_1
mapping_indices = np.argmax(coupling_matrix, axis=1)

# Compute the overall transport cost
overall_transport_cost = np.sum(coupling_matrix * cost_matrix)

# 2) Plot both datasets and the mapping
plt.figure(figsize=(8, 6))
plt.scatter(data_0[:, 0], data_0[:, 1], color='blue', label='Data 0')
plt.scatter(data_1[:, 0], data_1[:, 1], color='red', label='Data 1')

# Draw lines between matched points
for i, j in enumerate(mapping_indices):
    plt.plot([data_0[i, 0], data_1[j, 0]], [data_0[i, 1], data_1[j, 1]], color='gray', alpha=0.5)

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Optimal Transport Map between Two Datasets')
plt.legend()
plt.show()

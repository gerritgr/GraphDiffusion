import torch
from torch.utils.data import DataLoader, TensorDataset
import random


# Define a simple default train class
class VectorTrain():
    def __init__(self, node_feature_dim=1, batch_size = 1):
        super(VectorTrain, self).__init__()
        self.node_feature_dim = node_feature_dim
    
    def __call__(self, pipeline, data, epochs, *args, **kwargs):
        # Placeholder training logic
        return data
    
    def input_to_dataloader(self, input_data, batch_size, device):
        # Function to move a tensor or tensors in a list to the specified device
        def to_device(data):
            if isinstance(data, torch.Tensor):
                return data.to(device)
            elif isinstance(data, list):
                return [x.to(device) for x in data]
            return data
    
        if isinstance(input_data, DataLoader): #TODO fix
            return input_data

        # Check if input_data is a DataLoader
        if isinstance(input_data, DataLoader):
            # Check if DataLoader batch size matches
            if input_data.batch_size != batch_size:
                raise ValueError("Dataloader batch size does not match the specified batch size")

            # Check and move data in DataLoader to the correct device
            # Note: This creates a new DataLoader
            dataloader = DataLoader(
                dataset=TensorDataset(*[to_device(x) for x in input_data.dataset]),
                batch_size=batch_size,
                shuffle=True
            )
            return dataloader
        # Check if input_data is a list of tensors and create a DataLoader
        elif isinstance(input_data, list) and all(isinstance(x, torch.Tensor) for x in input_data):
            dataset = TensorDataset(*to_device(input_data))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return dataloader
        # Check if input_data is a single tensor and create a DataLoader
        elif isinstance(input_data, torch.Tensor):
            dataset = TensorDataset(to_device(input_data))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return dataloader
        else:
            raise TypeError("Input data must be a DataLoader, a list of tensors, or a single tensor")

        return dataloader  # or other relevant return value
    
    def __call__(self, pipeline, input_data, epochs=100, *args, **kwargs):
        dataloader = self.input_to_dataloader(input_data, pipeline.batch_size, pipeline.device)
        model = pipeline.get_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            for batch in dataloader:
                t = random.random()
                optimizer.zero_grad()
                batch_with_noise = pipeline.degradation(batch, t)
                batch_reconstructiond = pipeline.reconstruction(batch_with_noise, t)
                loss = pipeline.loss(batch_with_noise, batch_reconstructiond)
                print("loss", loss)
                loss.backward()
                optimizer.step()

        return dataloader


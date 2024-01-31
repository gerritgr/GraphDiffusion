import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())


import torch
from torch.utils.data import DataLoader, TensorDataset
import random


# Define a simple default train class
class VectorTrain:
    def __init__(self, pipeline):
        #super(VectorTrain, self).__init__()
        pass

    @staticmethod
    def input_to_dataloader(self, input_data, device, batch_size_default=1):
        # Function to move a tensor or tensors in a list to the specified device
        def to_device(data):
            if isinstance(data, torch.Tensor):
                return data.to(device)
            elif isinstance(data, list):
                return [x.to(device) for x in data]
            return data

        if isinstance(input_data, DataLoader):
            first_batch = next(iter(input_data))
            if first_batch.device == device:
                return input_data
            else:
                return DataLoader(
                    dataset=[to_device(x) for x in input_data.dataset],
                    batch_size=input_data.batch_size,
                    shuffle=True,
                )

        # Check if input_data is a list of tensors and create a DataLoader
        elif isinstance(input_data, list) and all(
            isinstance(x, torch.Tensor) for x in input_data
        ):
            dataloader = DataLoader(
                dataset=[to_device(x) for x in input_data],
                batch_size=batch_size_default,
                shuffle=True,
            )
            return dataloader
        # Check if input_data is a single tensor and create a DataLoader
        elif isinstance(input_data, torch.Tensor):
            dataset = [to_device(input_data)]
            dataloader = DataLoader(
                dataset, batch_size=batch_size_default, shuffle=True
            )
            return dataloader
        else:
            raise TypeError(
                "Input data must be a DataLoader, a list of tensors, or a single tensor"
            )

        return dataloader  # or other relevant return value

    def __call__(self, input_data, epochs=100, alpha=0.1, pipeline=None, *args, **kwargs):
        dataloader = VectorTrain.input_to_dataloader(input_data, pipeline.device)
        model = pipeline.get_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create a tqdm progress bar for the epochs
        pbar = tqdm(range(epochs), desc="Epoch: 0, Loss: N/A")

        # Initialize moving average loss
        moving_average_loss = None

        for epoch in pbar:
            total_loss = 0
            for batch in dataloader:
                t = torch.rand(batch.shape[0])  # random.random()
                optimizer.zero_grad()
                batch_with_noise = pipeline.degradation(batch, t)
                batch_reconstructed = pipeline.reconstruction(batch_with_noise, t)
                loss = pipeline.distance(batch, batch_reconstructed)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / len(
                dataloader
            )  # assumes mean reduction for loss
            # Update moving average loss
            if moving_average_loss is None:
                moving_average_loss = (
                    average_loss  # Initialize with the first epoch's average loss
                )
            else:
                moving_average_loss = (
                    alpha * average_loss + (1 - alpha) * moving_average_loss
                )

            # Update the progress bar description with the current epoch and moving average loss
            pbar.set_description(f"Epoch: {epoch + 1}, Loss: {moving_average_loss:.4f}")

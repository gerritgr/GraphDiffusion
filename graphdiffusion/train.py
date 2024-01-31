import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())


import torch
from torch.utils.data import DataLoader, TensorDataset
import random


class VectorTrain:
    """
    A class for training a model using a specified pipeline that encapsulates the model, degradation, reconstruction, and distance calculation.
    
    Attributes:
        pipeline (object): An object encapsulating the model, degradation, reconstruction, and distance calculation.
    """

    def __init__(self):
        """
        Initializes the VectorTrain class.

        Args:
            pipeline (object): An object representing the processing pipeline including model, degradation, reconstruction, and distance calculation.
        """
        pass

    @staticmethod
    def input_to_dataloader(input_data, device, batch_size_default=1):
        """
        Converts input data into a DataLoader, ensuring it is on the correct device.

        Args:
            input_data (DataLoader, list of Tensors, or Tensor): The input data to be converted.
            device (torch.device): The device to which the data should be moved.
            batch_size_default (int): The batch size to use if creating a new DataLoader. Default is 1.

        Returns:
            DataLoader: A DataLoader containing the input data on the specified device.

        Raises:
            TypeError: If the input data is not a DataLoader, list of Tensors, or a single Tensor.
        """
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
        elif isinstance(input_data, list) and all(isinstance(x, torch.Tensor) for x in input_data):
            dataloader = DataLoader(
                dataset=[to_device(x) for x in input_data],
                batch_size=batch_size_default,
                shuffle=True,
            )
            return dataloader
        # Check if input_data is a single tensor and create a DataLoader
        elif isinstance(input_data, torch.Tensor):
            dataset = [to_device(input_data)]
            dataloader = DataLoader(dataset, batch_size=batch_size_default, shuffle=True)
            return dataloader
        else:
            raise TypeError("Input data must be a DataLoader, a list of tensors, or a single tensor")


    def __call__(self, data, epochs=100, alpha=0.1, pipeline=None, *args, **kwargs):
        """
        Train the model using the provided input data and pipeline.

        Args:
            data (DataLoader, list of Tensors, or Tensor): The input data for training.
            epochs (int): The number of epochs to train for. Default is 100.
            alpha (float): The weight for the moving average of the loss. Default is 0.1.

        Returns:
            None
        """
        assert pipeline is not None, "pipeline must be provided"
        self.pipeline = pipeline
        dataloader = VectorTrain.input_to_dataloader(input_data=data, device=self.pipeline.device)
        model = self.pipeline.get_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initialize tqdm progress bar
        pbar = tqdm(range(epochs), desc="Epoch: 0, Loss: N/A")

        # Initialize moving average loss
        moving_average_loss = None

        for epoch in pbar:
            total_loss = 0
            for batch in dataloader:
                # Generate random tensor 't' for degradation
                t = torch.rand(batch.shape[0], device=self.pipeline.device)

                optimizer.zero_grad()
                
                # Apply degradation and reconstruction from the pipeline
                batch_with_noise = self.pipeline.degradation(batch, t)
                batch_reconstructed = self.pipeline.reconstruction(batch_with_noise, t)
                
                # Compute loss using the pipeline's distance method
                loss = self.pipeline.distance(batch, batch_reconstructed)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Calculate average loss for the epoch
            average_loss = total_loss / len(dataloader)

            # Update moving average loss
            if moving_average_loss is None:
                moving_average_loss = average_loss  # Initialize with the first epoch's average loss
            else:
                moving_average_loss = alpha * average_loss + (1 - alpha) * moving_average_loss

            # Update the progress bar description
            pbar.set_description(f"Epoch: {epoch + 1}, Loss: {moving_average_loss:.4f}")
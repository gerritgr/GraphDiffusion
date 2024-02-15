import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "imports.py")
with open(file_path, "r") as file:
    exec(file.read())


import torch
from torch.utils.data import DataLoader, TensorDataset
import random


def model_test(pipeline, dataloader):  # cannot name it test_model due to test suit
    total_loss = 0
    pipeline.get_model().eval()
    with torch.no_grad():  # redundant?
        for batch in dataloader:
            t = torch.rand(batch.shape[0], device=batch.device)
            batch_with_noise = pipeline.degradation(batch, t)
            batch_reconstructed = pipeline.reconstruction(batch_with_noise, t)
            loss = pipeline.distance(batch, batch_reconstructed)
            total_loss += loss.item()

    pipeline.get_model().train()
    average_loss_test = total_loss / len(dataloader)
    return average_loss_test


def train_epoch(dataloader, pipeline, optimizer):
    model = pipeline.get_model()
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = pipeline.preprocess(batch) 
        
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch, label = batch

        # Generate random tensor 't' for degradation
        t = torch.rand(batch.shape[0], device=pipeline.device)

        optimizer.zero_grad()

        # Apply degradation and reconstruction from the pipeline
        batch_with_noise = pipeline.degradation(batch, t)
        batch_reconstructed = pipeline.reconstruction(batch_with_noise, t)

        # Compute loss using the pipeline's distance method
        loss = pipeline.distance(batch, batch_reconstructed)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate average loss for the epoch
    average_loss = total_loss / len(dataloader)
    return average_loss


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
            if isinstance(first_batch, list) or isinstance(first_batch, tuple):
                first_batch = first_batch[0]

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

    def __call__(self, data, pipeline, data_test=None, epochs=100, alpha=0.1, load_optimizer_state=True, init_learning_rate=1e-3, *args, **kwargs):
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
        dataloader_train = VectorTrain.input_to_dataloader(input_data=data, device=self.pipeline.device)
        dataloader_test = VectorTrain.input_to_dataloader(input_data=data_test, device=self.pipeline.device) if data_test is not None else None
        model = self.pipeline.get_model()
        model.train()
        if load_optimizer_state:
            try:
                optimizer = pipeline.optimizer
                pipeline.debug("Optimizer found in pipeline.")
            except AttributeError:
                optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
                pipeline.optimizer = optimizer
                pipeline.debug("Created new optimizer.")
            try:
                optimizer.load_state_dict(pipeline.optimizer_state_dict)
                pipeline.debug("Loaded optimizer state.")
            except:
                pipeline.debug("Could not load optimizer state dict.")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
            pipeline.optimizer = optimizer
            pipeline.debug("Created new optimizer.")

        # Initialize tqdm progress bar
        pbar = tqdm(range(epochs), desc="Epoch: 0, Loss: N/A, Test Loss: N/A") if data_test is not None else tqdm(range(epochs), desc="Epoch: 0, Loss: N/A")

        # Initialize moving average loss
        moving_average_loss = None
        moving_average_loss_test = None
        average_loss = np.nan

        for epoch in pbar:
            # TODO: if epoch is zero or epoch is the last epoch, or epoch is dividable by 10: compute the loss using dataloader_test and save it as total_loss_test
            if dataloader_test is not None and (epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0):
                average_loss_test = model_test(self.pipeline, dataloader_test)
                pipeline.debug(f"Test loss: {average_loss_test:.5f}, Epoch: {epoch}")
                pipeline.debug(f"Train loss: {average_loss:.5f}, Epoch: {epoch}")

            average_loss = train_epoch(dataloader_train, self.pipeline, optimizer)

            # Update moving average loss
            if moving_average_loss is None:
                moving_average_loss = average_loss  # Initialize with the first epoch's average loss
            else:
                moving_average_loss = alpha * average_loss + (1 - alpha) * moving_average_loss

            if data_test is not None:
                if moving_average_loss_test is None:
                    moving_average_loss_test = average_loss_test  # Initialize with the first epoch's average loss
                else:
                    moving_average_loss_test = alpha * average_loss_test + (1 - alpha) * moving_average_loss_test

                # Update the progress bar description
                pbar.set_description(f"Epoch: {epoch + 1}, Loss: {moving_average_loss:.4f}, Test Loss: {moving_average_loss_test:.4f}")
            else:
                pbar.set_description(f"Epoch: {epoch + 1}, Loss: {moving_average_loss:.4f}")

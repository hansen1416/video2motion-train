import os

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MediapipeTransferLinear import MediapipeTransferLinear
from dataset import (
    DATA_DIR,
    MediapipeDataset,
    FilewiseShuffleSampler,
    fetch_datatset_info,
)


if __name__ == "__main__":

    inputs_dir_train = os.path.join(
        os.path.dirname(__file__), "dataset", "data", "inputs"
    )
    outputs_dir_train = os.path.join(
        os.path.dirname(__file__), "dataset", "data", "outputs"
    )

    # Fetch the dataset info
    indices_to_file_index_train, data_indices_in_files_train = fetch_datatset_info(
        inputs_dir_train, outputs_dir_train
    )

    train_dataset = MediapipeDataset(
        inputs_dir_train,
        outputs_dir_train,
        indices_to_file_index_train,
        data_indices_in_files_train,
        device="cpu",
    )

    inputs_dir_test = os.path.join(
        os.path.dirname(__file__),
        "dataset",
        "data",
        "test",
        "inputs",
    )
    outputs_dir_test = os.path.join(
        os.path.dirname(__file__),
        "dataset",
        "data",
        "test",
        "outputs",
    )

    # Fetch the dataset info
    indices_to_file_index_test, data_indices_in_files_test = fetch_datatset_info(
        inputs_dir_test, outputs_dir_test
    )

    test_dataset = MediapipeDataset(
        inputs_dir_test,
        outputs_dir_test,
        indices_to_file_index_test,
        data_indices_in_files_test,
        device="cpu",
    )

    # get the first item from the dataset

    print(len(train_dataset), len(test_dataset))

    # Assuming you have your CustomDataset class defined

    # Hyperparameters (adjust based on your needs)
    learning_rate = 0.001
    epochs = 500

    model = MediapipeTransferLinear()

    # Define loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction="none")  # Mean squared error for regression
    loss_fn2 = torch.nn.MSELoss(reduction="mean")  # Mean squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    BTACH_SIZE = 8

    # Create data loaders (replace with your actual data paths)
    train_loader = DataLoader(train_dataset, batch_size=BTACH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BTACH_SIZE, shuffle=False)

    # print(train_loader)

    writer = SummaryWriter()

    # Train the model
    for epoch in range(epochs):
        for i, (features, targets) in enumerate(train_loader):  # Ignore labels for now
            # Forward pass
            outputs = model(features)

            # # reshape the outputs from (bacth_size, 66) to (batch_size, 22, 3)
            # outputs = outputs.reshape(-1, 22, 3)
            # targets = targets.reshape(-1, 22, 3)

            # euler2vector(outputs)

            # Calculate loss
            loss = loss_fn(
                outputs, targets
            )  # Compare outputs with input data (assume labels unavailable)

            # print(loss)

            # get the sqrt of the sum of the squares `loss_mid`
            loss_result = torch.sqrt(torch.sum(torch.sum(loss, dim=2) ** 2))

            # print(loss_result)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss_result.backward()
            optimizer.step()

            writer.add_scalar(
                "Loss/train", loss_result.item(), epoch * len(train_loader) + i
            )  # Log train loss

            # Validation (optional, replace with your validation logic)
            if (i + 1) % 10 == 0:  # Validate every 100 batches
                with torch.no_grad():
                    test_loss_value = 0.0
                    test_acc = 0.0
                    for test_features, test_targets in test_loader:
                        test_outputs = model(test_features)

                        test_loss = loss_fn(test_outputs, test_targets)
                        test_loss_result = torch.sqrt(
                            torch.sum(torch.sum(test_loss, dim=2) ** 2)
                        )

                        test_loss_value += test_loss_result.item()

                    test_loss_value /= len(test_loader)

                    writer.add_scalar(
                        "Loss/test", test_loss_value, epoch * len(train_loader) + i
                    )

                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss_result.item():.4f}, Test Loss: {test_loss_value:.4f}"
                )

        # Print training progress (optional)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_result.item():.4f}")

        #     break
        # break

    writer.close()

    # save the model to local file
    torch.save(model.state_dict(), os.path.join("models", "model.pth"))

# # Evaluate the model (assuming you have validation data)
# with torch.no_grad():
#     for data, _ in test_loader:
#         outputs = model(data)
#         # Calculate and print loss or other evaluation metrics
#         ...

# print("Training complete!")

import os

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MediapipeTransferLinear import MediapipeTransferLinear
from MediapipeDataset import get_dataloader


if __name__ == "__main__":

    from constants import TRAIN_DATASET_DIR, TEST_DATASET_DIR

    inputs_dir_train = os.path.join(TRAIN_DATASET_DIR, "inputs")
    outputs_dir_train = os.path.join(TRAIN_DATASET_DIR, "outputs")

    inputs_dir_test = os.path.join(TEST_DATASET_DIR, "inputs")
    outputs_dir_test = os.path.join(TEST_DATASET_DIR, "outputs")

    train_loader = get_dataloader(inputs_dir_train, outputs_dir_train)
    test_loader = get_dataloader(inputs_dir_test, outputs_dir_test)

    print(len(train_loader), len(test_loader))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Hyperparameters (adjust based on your needs)
    learning_rate = 0.001
    epochs = 500

    model = MediapipeTransferLinear()
    model.to(device)

    # Define loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction="none").to(device)
    loss_fn2 = torch.nn.MSELoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    # Train the model
    for epoch in range(epochs):
        for i, (features, targets) in enumerate(train_loader):  # Ignore labels for now
            # Forward pass
            outputs = model(features)

            # Calculate loss
            loss = loss_fn(
                outputs, targets
            )  # Compare outputs with input data (assume labels unavailable)

            # get the sqrt of the sum of the squares `loss_mid`
            loss_result = torch.sqrt(torch.sum(torch.sum(loss, dim=2) ** 2))

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

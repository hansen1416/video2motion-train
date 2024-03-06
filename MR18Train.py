import os
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MR18Dataset import MR18Dataset
from MR18Model import MR18Model

if __name__ == "__main__":

    from constants import (
        MEDIAPIPE_JOINED_DIR,
        RESNET_JOINED_DIR,
        ANIM_EULER_JOINED_DIR,
        CHECKPOINT_DIR,
    )

    dataset = MR18Dataset(
        MEDIAPIPE_JOINED_DIR, RESNET_JOINED_DIR, ANIM_EULER_JOINED_DIR
    )

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(len(train_loader), len(test_loader))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters (adjust based on your needs)
    learning_rate = 0.001
    epochs = 500

    model = MR18Model()
    model.to(device)

    # Define loss function and optimizer
    loss_fn = torch.nn.MSELoss(reduction="none").to(device)
    loss_fn2 = torch.nn.MSELoss(reduction="mean").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    # Train the model
    for epoch in range(epochs):
        for i, (mediapipe_input, resnet_input, anim_euler_target) in enumerate(
            train_loader
        ):  # Ignore labels for now
            # Forward pass
            outputs = model(mediapipe_input, resnet_input)

            # Calculate loss
            loss = loss_fn(
                outputs, anim_euler_target
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

            if i and (i % 1000 == 0):
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_loader)}, Loss: {loss_result.item():.4f}"
                )

        # Validation (optional, replace with your validation logic)
        with torch.no_grad():
            test_loss_value = 0.0
            test_acc = 0.0
            for (
                mediapipe_input_test,
                resnet_input_test,
                anim_euler_target_test,
            ) in test_loader:
                test_outputs = model(mediapipe_input_test, resnet_input_test)

                test_loss = loss_fn(test_outputs, anim_euler_target_test)
                test_loss_result = torch.sqrt(
                    torch.sum(torch.sum(test_loss, dim=2) ** 2)
                )

                test_loss_value += test_loss_result.item()

            test_loss_value /= len(test_loader)

        writer.add_scalar("Loss/test", test_loss_value, (epoch + 1) * len(train_loader))

        print(
            f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss_result.item():.4f}, Test Loss: {test_loss_value:.4f}"
        )

        # every 5 epochs, save the model to local file
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"{model.__class__.__name__}_{epoch}.pth"),
            )

    writer.close()

import os
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from MediapipeResnet18JoinedDataset import MediapipeResnet18JoinedDataset


class WeightedCombineLayer(nn.Module):
    def __init__(self):
        super(WeightedCombineLayer, self).__init__()
        # Initialize weights with 0.5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize weights with 0.5
        self.weight_inp1 = torch.nn.Parameter(torch.full((1, 66), 0.5)).to(device)
        self.weight_inp2 = torch.nn.Parameter(torch.full((1, 66), 0.5)).to(device)

    def forward(self, input1, input2):
        return input1 * self.weight_inp2 + input2 * self.weight_inp2


class MediapipeResnet18JoinedModel(nn.Module):
    def __init__(self) -> None:
        super(MediapipeResnet18JoinedModel, self).__init__()

        self.res_dp1 = nn.Dropout(0.25)
        self.res_fc1 = nn.Linear(512, 256)
        self.res_dp2 = nn.Dropout(0.25)
        self.res_fc2 = nn.Linear(256, 128)
        self.res_dp3 = nn.Dropout(0.25)
        self.res_fc3 = nn.Linear(128, 66)

        self.mp_fc1 = nn.Linear(99, 128)
        self.mp_fc2 = nn.Linear(128, 256)
        self.mp_fc3 = nn.Linear(256, 128)
        self.mp_fc4 = nn.Linear(128, 66)

    def forward(self, mediapipe_input, resnet_input):

        mediapipe_input = self.mp_fc1(mediapipe_input)
        mediapipe_input = self.mp_fc2(mediapipe_input)
        mediapipe_input = self.mp_fc3(mediapipe_input)
        mediapipe_input = self.mp_fc4(mediapipe_input)
        # now the shape pf mediapipe_input is (batch_size, 66)

        resnet_input = self.res_dp1(resnet_input)
        resnet_input = self.res_fc1(resnet_input)
        resnet_input = self.res_dp2(resnet_input)
        resnet_input = self.res_fc2(resnet_input)
        resnet_input = self.res_dp3(resnet_input)
        resnet_input = self.res_fc3(resnet_input)
        # clamp the values to be between -pi and pi
        resnet_input = torch.clamp(resnet_input, -np.pi, np.pi)
        # now the shape of resnet_input is (batch_size, 66)

        # apply the weighted combine layer, the output shape is (batch_size, 66)
        combined_output = WeightedCombineLayer()(mediapipe_input, resnet_input)

        # Reshape to (batch_size, 22, 3), so that we can use it as input to the loss function
        output = combined_output.reshape(-1, 22, 3)

        return output


if __name__ == "__main__":

    from constants import MEDIAPIPE_JOINED_DIR, RESNET_JOINED_DIR, ANIM_EULER_JOINED_DIR

    dataset = MediapipeResnet18JoinedDataset(
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

    model = MediapipeResnet18JoinedModel()
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
            torch.save(model.state_dict(), os.path.join("models", f"model_{epoch}.pth"))

    writer.close()

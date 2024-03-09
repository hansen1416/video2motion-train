import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Datensatz
from model import Model


def train(
    mediapipe_data_file,
    anim_euler_data_file,
    checkpoint_dir="./checkpoints",
    logdir="./runs",
    write_log=True,
    save_every=10,
    batch_size=128,
):
    dataset = Datensatz(
        mediapipe_data_file,
        anim_euler_data_file,
    )

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(len(train_loader), len(test_loader))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters (adjust based on your needs)
    learning_rate = 0.001
    epochs = 500

    model = Model()
    model.to(device)

    # Define loss function and optimizer

    loss_fn1 = torch.nn.MSELoss(reduction="mean").to(device)
    loss_fn2 = torch.nn.MSELoss(reduction="mean").to(device)
    loss_fn3 = torch.nn.MSELoss(reduction="mean").to(device)
    loss_fn4 = torch.nn.MSELoss(reduction="mean").to(device)
    loss_fn5 = torch.nn.MSELoss(reduction="mean").to(device)
    loss_fn6 = torch.nn.MSELoss(reduction="mean").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=logdir)

    # Train the model
    for epoch in range(epochs):
        for i, (mediapipe_input, anim_euler_target) in enumerate(
            train_loader
        ):  # Ignore labels for now
            # Forward pass
            (
                hip_spine_pred,
                neck_head_pred,
                right_arm_pred,
                left_arm_pred,
                right_leg_pred,
                left_leg_pred,
            ) = model(mediapipe_input)

            hip_spine_true = anim_euler_target[:, [0, 1, 2, 3], :].view(-1, 12)
            neck_head_true = anim_euler_target[:, [4, 5], :].view(-1, 6)
            right_arm_true = anim_euler_target[:, [6, 7, 8, 9], :].view(-1, 12)
            left_arm_true = anim_euler_target[:, [10, 11, 12, 13], :].view(-1, 12)
            right_leg_true = anim_euler_target[:, [14, 15, 16, 17], :].view(-1, 12)
            left_leg_true = anim_euler_target[:, [18, 19, 20, 21], :].view(-1, 12)

            loss1 = loss_fn1(hip_spine_pred, hip_spine_true)
            loss2 = loss_fn2(neck_head_pred, neck_head_true)
            loss3 = loss_fn3(right_arm_pred, right_arm_true)
            loss4 = loss_fn4(left_arm_pred, left_arm_true)
            loss5 = loss_fn5(right_leg_pred, right_leg_true)
            loss6 = loss_fn6(left_leg_pred, left_leg_true)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss1.backward()
            loss2.backward()
            loss3.backward()
            loss4.backward()
            loss5.backward()
            loss6.backward()
            optimizer.step()

            if write_log:
                writer.add_scalar(
                    "Loss1/train", loss1.item(), epoch * len(train_loader) + i
                )  # Log train loss
                writer.add_scalar(
                    "Loss2/train", loss2.item(), epoch * len(train_loader) + i
                )
                writer.add_scalar(
                    "Loss3/train", loss3.item(), epoch * len(train_loader) + i
                )
                writer.add_scalar(
                    "Loss4/train", loss4.item(), epoch * len(train_loader) + i
                )
                writer.add_scalar(
                    "Loss5/train", loss5.item(), epoch * len(train_loader) + i
                )
                writer.add_scalar(
                    "Loss6/train", loss6.item(), epoch * len(train_loader) + i
                )

            if i and (i % 100 == 0):
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_loader)}, \
                        Loss: {loss1.item():.4f}, {loss2.item():.4f}, {loss3.item():.4f}, {loss4.item():.4f}, {loss5.item():.4f}, {loss6.item():.4f}"
                )

        # # Validation (optional, replace with your validation logic)
        # with torch.no_grad():
        #     test_loss_value = 0.0

        #     for (
        #         mediapipe_input_test,
        #         anim_euler_target_test,
        #     ) in test_loader:
        #         test_outputs = model(mediapipe_input_test)

        #         test_loss = loss_fn(test_outputs, anim_euler_target_test)
        #         test_loss_result = torch.sqrt(
        #             torch.sum(torch.sum(test_loss, dim=2) ** 2)
        #         )

        #         test_loss_value += test_loss_result.item()

        #     test_loss_value /= len(test_loader)

        # if write_log:
        #     writer.add_scalar(
        #         "Loss/test", test_loss_value, (epoch + 1) * len(train_loader)
        #     )

        print(
            f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, \
                Loss: {loss1.item():.4f}, {loss2.item():.4f}, {loss3.item():.4f}, {loss4.item():.4f}, {loss5.item():.4f}, {loss6.item():.4f}"
        )

        # every 5 epochs, save the model to local file
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f"{model.__class__.__name__}_{epoch}.pth"),
            )

    writer.close()


if __name__ == "__main__":

    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from constants import MEDIAPIPE_JOINED_DIR, ANIM_EULER_JOINED_DIR, CHECKPOINT_DIR

    train(
        os.path.join(MEDIAPIPE_JOINED_DIR, "joined.npy"),
        os.path.join(ANIM_EULER_JOINED_DIR, "joined.npy"),
        checkpoint_dir=CHECKPOINT_DIR,
    )

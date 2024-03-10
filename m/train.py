import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

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

    print("using device ", device)

    model = Model()
    model.to(device)

    model.train()

    # use a leaning rate scheduler
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=300
    )
    epochs = 500

    # Define loss function and optimizer

    lossfn_hip_spine = torch.nn.MSELoss(reduction="mean").to(device)
    lossfn_neck_head = torch.nn.MSELoss(reduction="mean").to(device)
    lossfn_right_arm = torch.nn.MSELoss(reduction="mean").to(device)
    lossfn_right_hand = torch.nn.MSELoss(reduction="mean").to(device)
    lossfn_left_arm = torch.nn.MSELoss(reduction="mean").to(device)
    lossfn_left_hand = torch.nn.MSELoss(reduction="mean").to(device)
    lossfn_right_leg = torch.nn.MSELoss(reduction="mean").to(device)
    lossfn_right_foot = torch.nn.MSELoss(reduction="mean").to(device)
    lossfn_left_leg = torch.nn.MSELoss(reduction="mean").to(device)
    lossfn_left_foot = torch.nn.MSELoss(reduction="mean").to(device)

    lossfn_test = torch.nn.MSELoss(reduction="mean").to(device)

    writer = SummaryWriter(log_dir=logdir)

    # Train the model
    for epoch in range(epochs):

        train_loss_value = 0.0

        for i, (mediapipe_input, anim_euler_target) in enumerate(
            train_loader
        ):  # Ignore labels for now
            # Forward pass
            (
                hip_spine_pred,
                neck_head_pred,
                right_arm_pred,
                right_hand_pred,
                left_arm_pred,
                left_hand_pred,
                right_leg_pred,
                right_foot_pred,
                left_leg_pred,
                left_foot_pred,
            ) = model(mediapipe_input)

            hip_spine_true = anim_euler_target[:, [0, 1, 2, 3], :].view(-1, 12)
            neck_head_true = anim_euler_target[:, [4, 5], :].view(-1, 6)
            right_arm_true = anim_euler_target[:, [6, 7, 8], :].view(-1, 9)
            right_hand_true = anim_euler_target[:, [9], :].view(-1, 3)
            left_arm_true = anim_euler_target[:, [10, 11, 12], :].view(-1, 9)
            left_hand_true = anim_euler_target[:, [13], :].view(-1, 3)
            right_leg_true = anim_euler_target[:, [14, 15], :].view(-1, 6)
            right_foot_true = anim_euler_target[:, [16, 17], :].view(-1, 6)
            left_leg_true = anim_euler_target[:, [18, 19], :].view(-1, 6)
            left_foot_true = anim_euler_target[:, [20, 21], :].view(-1, 6)

            loss_hip_spine = lossfn_hip_spine(hip_spine_pred, hip_spine_true)
            loss_neck_head = lossfn_neck_head(neck_head_pred, neck_head_true)
            loss_right_arm = lossfn_right_arm(right_arm_pred, right_arm_true)
            loss_right_hand = lossfn_right_hand(right_hand_pred, right_hand_true)
            loss_left_arm = lossfn_left_arm(left_arm_pred, left_arm_true)
            loss_left_hand = lossfn_left_hand(left_hand_pred, left_hand_true)
            loss_right_leg = lossfn_right_leg(right_leg_pred, right_leg_true)
            loss_right_foot = lossfn_right_foot(right_foot_pred, right_foot_true)
            loss_left_leg = lossfn_left_leg(left_leg_pred, left_leg_true)
            loss_left_foot = lossfn_left_foot(left_foot_pred, left_foot_true)

            # Backward pass and optimize
            optimizer.zero_grad()

            loss_hip_spine.backward()
            loss_neck_head.backward()
            loss_right_arm.backward()
            loss_right_hand.backward()
            loss_left_arm.backward()
            loss_left_hand.backward()
            loss_right_leg.backward()
            loss_right_foot.backward()
            loss_left_leg.backward()
            loss_left_foot.backward()

            optimizer.step()

            train_loss_value += (
                loss_hip_spine.item()
                + loss_neck_head.item()
                + loss_right_arm.item()
                + loss_right_hand.item()
                + loss_left_arm.item()
                + loss_left_hand.item()
                + loss_right_leg.item()
                + loss_right_foot.item()
                + loss_left_leg.item()
                + loss_left_foot.item()
            ) / 10

            if i and (i % 100 == 0):
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_loader)}, Loss: "
                    + f"hip_spine: {loss_hip_spine.item():.4f}, neck_head {loss_neck_head.item():.4f}, "
                    + f"right_arm {loss_right_arm.item():.4f}, right_hand {loss_right_hand.item():.4f}, "
                    + f"left_arm {loss_left_arm.item():.4f}, left_hand {loss_left_hand.item():.4f}, "
                    + f"right_leg {loss_right_leg.item():.4f}, right_foot {loss_right_foot.item():.4f}, "
                    + f"left_leg {loss_left_leg.item():.4f}, left_foot {loss_left_foot.item():.4f}"
                )

        scheduler.step()

        if write_log:
            writer.add_scalar(
                "Loss/train", train_loss_value / len(train_loader), (epoch + 1)
            )

        # Validation (optional, replace with your validation logic)
        with torch.no_grad():
            test_loss_value = 0.0

            for (
                mediapipe_input_test,
                anim_euler_target_test,
            ) in test_loader:
                (
                    hip_spine,
                    neck_head,
                    right_arm,
                    right_hand,
                    left_arm,
                    left_hand,
                    right_leg,
                    right_foot,
                    left_leg,
                    left_foot,
                ) = model(mediapipe_input_test)

                test_outputs = torch.cat(
                    (
                        hip_spine,
                        neck_head,
                        right_arm,
                        right_hand,
                        left_arm,
                        left_hand,
                        right_leg,
                        right_foot,
                        left_leg,
                        left_foot,
                    ),
                    1,
                )

                test_outputs = test_outputs.reshape(-1, 22, 3)

                test_loss = lossfn_test(test_outputs, anim_euler_target_test)

                test_loss_value += test_loss.item()

        if write_log:
            writer.add_scalar(
                "Loss/test", test_loss_value / len(test_loader), (epoch + 1)
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
        save_every=1,
    )

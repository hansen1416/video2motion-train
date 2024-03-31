import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from lstm import Seq2SeqLSTM
from dataset import Datensatz


def train(
    model: nn.Module,
    checkpoint_dir="./checkpoints",
    pretrained_checkpoint=None,
    logdir="./runs",
    write_log=True,
    save_every=10,
    batch_size=512,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = Datensatz()

    # Split the dataset into train and test sets
    train_size = int(0.9 * len(ds))
    test_size = len(ds) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        ds, [train_size, test_size]
    )

    batch_size = 32
    seq_length = 29

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"train size: {len(train_loader)}, test size: {len(test_loader)}")

    start_epoch = 0

    if pretrained_checkpoint:

        last_epoch = int(pretrained_checkpoint.split("_")[-1].split(".")[0])

        model.load_state_dict(torch.load(pretrained_checkpoint, map_location=device))

        print(
            "load pretrained model from ",
            pretrained_checkpoint,
            " at epoch ",
            last_epoch,
        )

        start_epoch = last_epoch + 1

    model.to(device)

    model.train()

    # use a leaning rate scheduler
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=300
    )
    criterion = nn.MSELoss().to(device)

    writer = SummaryWriter(log_dir=logdir)

    num_epochs = 500

    best_test_loss = 1000

    for epoch in range(start_epoch, start_epoch + num_epochs):

        avg_tran_loss = 0.0

        for i, (input_tensor, target_tensor) in enumerate(train_loader):
            # the last batch might not be a full batch, skip it
            if input_tensor.shape[0] != batch_size:
                continue

            # print(input_tensor.shape, target_tensor.shape)

            optimizer.zero_grad()
            output = model(input_tensor.view(batch_size, seq_length, -1))
            # output = output.squeeze()
            loss = criterion(output, target_tensor.view(batch_size, seq_length, -1))
            loss.backward()
            optimizer.step()

            avg_tran_loss += loss.item()

        avg_tran_loss /= len(train_loader)

        if write_log:
            writer.add_scalar("Loss/train", avg_tran_loss, epoch)

        print(f"Epoch {epoch}, train loss: {avg_tran_loss}")

        if epoch and epoch % save_every == 0:

            with torch.no_grad():
                avg_test_loss = 0.0

                for input_tensor, target_tensor in test_loader:
                    if input_tensor.shape[0] != batch_size:
                        continue

                    output = model(input_tensor.view(batch_size, seq_length, -1))
                    loss = criterion(
                        output, target_tensor.view(batch_size, seq_length, -1)
                    )
                    avg_test_loss += loss.item()

                avg_test_loss /= len(test_loader)

                if write_log:
                    writer.add_scalar("Loss/test", avg_test_loss, (epoch + 1))

                print(f"Epoch {epoch}, test loss: {avg_test_loss}")

                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    torch.save(
                        model.state_dict(),
                        f"{checkpoint_dir}/{model.__class__.__name__}_{epoch}.pth",
                    )

                    print(f"save model at epoch {epoch} with test loss {avg_test_loss}")

    writer.close()


if __name__ == "__main__":

    model = Seq2SeqLSTM(input_size=17 * 3, hidden_size=64, num_layers=2)

    train(model, pretrained_checkpoint=os.path.join("checkpoints", "model_280.pth"))

import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

from dataset import Datensatz

load_dotenv()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v):
        """
        Args:
            d_in: int, embedding_size, we use 3 as the embedding size, only for demonstration,
                in practice, use a larger embedding size. eg. Llama 2 utilizes embedding sizes of 4,096

            d_out_kq: int, the number of elements in the query and key vectors, d_q = d_k
                Since we are computing the dot-product between the query and key vectors,
                these two vectors have to contain the same number of elements (d_q = d_k) `d_out_kq`

            d_out_v: int, the number of elements in the value vector v(i),
                In many LLMs, we use the same size for the value vectors such that d_q = d_k = d_v.
                However, the number of elements in the value vector v(i),
                which determines the size of the resulting context vector, can be arbitrary.
        """
        super().__init__()
        self.d_out_kq = d_out_kq

        # (embedding_size, d_out_kq)
        # self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_query = nn.Linear(d_in, d_out_kq)
        # (embedding_size, d_out_kq)
        # self.W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key = nn.Linear(d_in, d_out_kq)
        # (embedding_size, d_out_v)
        # self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))
        self.W_value = nn.Linear(d_in, d_out_v)

    def forward(self, x):

        # each item in `queries` is the queries weights for each word in the sentence
        # represents what information a specific element in the sequence needs from others. therefore keys.T
        queries = self.W_query(x)

        # each item in `keys` is the keys weights for each word in the sentence
        # represents what information each element in the sequence can provide.
        keys = self.W_key(x)

        # each item in `values` is the values weights for each word in the sentence
        # holds the actual information of each element.
        values = self.W_value(x)

        # (batch_size, sentence_length, embedding_size) @ (embedding_size, d_out_kq) = (batch_size, sentence_length, d_out_kq)
        # (batch_size, sentence_length, d_out_kq)
        # print(queries.shape, keys.shape, values.shape)

        # attention score $\omega_{i,j} = q^{(i)} k^{(j)}$
        # (batch_size, sentence_length, d_out_kq) @ (batch_size, d_out_kq, sentence_length) = (batch_size, sentence_length, sentence_length)
        attn_scores = queries @ keys.transpose(-2, -1)

        # to obtain the normalized attention weights, α (alpha),
        # by applying the softmax function. Additionally, 1/√{d_k} is used to scale $\omega$
        # before normalizing it through the softmax function
        # The scaling by d_k ensures that the Euclidean length of the weight vectors will be approximately in the same magnitude.
        # dim=-1. This ensures that the attention weights for each element (represented by rows in the tensor) sum up to 1.
        # (batch_size, sentence_length, sentence_length)
        attn_weights = torch.softmax(attn_scores / self.d_out_kq**0.5, dim=-1)

        # the context vector z^(i), which is an attention-weighted version of our original query input x^(i),
        # including all the other input elements as its context via the attention weights:
        # (batch_size, sentence_length, sentence_length) @ (batch_size, sentence_length, d_out_v) = (batch_size, sentence_length, d_out_v)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttention(nn.Module):
    """
    each attention head in multi-head attention can potentially learn to focus on different parts of the input sequence,
    capturing various aspects or relationships within the data.
    This diversity in representation is key to the success of multi-head attention.

    Multi-head attention can also be more efficient, especially in terms of parallel computation.
    Each head can be processed independently,
    making it well-suited for modern hardware accelerators like GPUs or TPUs that excel at parallel processing.

    eg. the 7B Llama 2 model uses 32 attention heads.
    """

    def __init__(self, input_seq_len, d_in, d_out_kq, d_out_v, num_heads):
        super().__init__()
        # each self-attention head will have its own set of weight matrices, they work in parallel
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v) for _ in range(num_heads)]
        )

        self.input_seq_len = input_seq_len
        self.d_in = d_in
        self.d_out_kq = d_out_kq
        self.d_out_v = d_out_v
        self.num_heads = num_heads

    def forward(self, x):

        # each head will produce a context vector
        # (num_heads, batch_size, sentence_length, d_out_v)
        head_outputs = torch.stack([head(x) for head in self.heads])

        # reshape the output to (batch_size, num_heads, sentence_length, d_out_v)
        head_outputs = head_outputs.permute(1, 0, 2, 3)

        return head_outputs


class MultiHeadLinearLayer(nn.Module):

    def __init__(
        self,
        num_heads,
        input_seq_len,
        d_out_v,
        hidden_size,
        output_size,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.input_seq_len = input_seq_len
        self.d_out_v = d_out_v
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_seq_len * d_out_v, hidden_size)
        self.fc2 = nn.Linear(hidden_size * num_heads, output_size)

    def forward(self, x):

        batch_size = x.shape[0]

        fc1_output = torch.zeros(batch_size, self.num_heads, self.hidden_size).to(
            device
        )
        # pass each head output through a linear layer
        for i in range(self.num_heads):
            xi = x[:, i, :, :].reshape(-1, self.input_seq_len * self.d_out_v)
            xi = self.fc1(xi)
            xi = F.tanh(xi)

            fc1_output[:, i, :] = xi

        # concatenate the output of each head
        output = fc1_output.reshape(batch_size, self.num_heads * self.hidden_size)

        output = self.fc2(output)

        return output


class DoubleAttention(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inner_seq_len = 17
        self.inner_num_heads = 4
        self.inner_d_in = 3
        self.inner_d_out_kq = 6
        self.inner_d_out_v = 8
        self.inner_linear_hidden = 256
        self.inner_linear_output = 64

        self.outter_seq_len = 30
        self.outter_d_in = self.inner_linear_output
        self.outter_d_out_kq = 64
        self.outter_d_out_v = 64
        self.outter_linear_hidden = 1024
        self.outter_linear_output = 36 * self.outter_seq_len
        self.outter_num_heads = 4

        self.ma_inner = MultiHeadAttention(
            self.inner_seq_len,
            self.inner_d_in,
            self.inner_d_out_kq,
            self.inner_d_out_v,
            self.inner_num_heads,
        )
        self.ml_inner = MultiHeadLinearLayer(
            self.inner_num_heads,
            self.inner_seq_len,
            self.inner_d_out_v,
            self.inner_linear_hidden,
            self.inner_linear_output,
        )

        self.ma_outter = MultiHeadAttention(
            self.outter_seq_len,
            self.outter_d_in,
            self.outter_d_out_kq,
            self.outter_d_out_v,
            self.outter_num_heads,
        )
        self.ml_outter = MultiHeadLinearLayer(
            self.outter_num_heads,
            self.outter_seq_len,
            self.outter_d_out_v,
            self.outter_linear_hidden,
            self.outter_linear_output,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_output = torch.zeros(
            x.shape[0], self.outter_seq_len, self.inner_linear_output
        ).to(self.device)

        # first learn the inner sequence separately
        for i in range(self.outter_seq_len):

            output = self.ma_inner(x[:, i, :, :])
            output = self.ml_inner(output)

            seq_output[:, i, :] = output

        # pass the inner sequence output to the outter sequence attention
        output = self.ma_outter(seq_output)

        output = self.ml_outter(output)

        # 12 bones and 3 coordinates
        return output.view(-1, self.outter_seq_len, 12, 3)


def train(
    train_loader,
    test_loader=None,
    pretrained=None,
    checkpoint_dir="./checkpoints",
    logdir="./runs",
    write_log=True,
    save_after_epoch=0,
):

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = DoubleAttention()

    start_epoch = 0

    if pretrained:

        last_epoch = int(pretrained.split("_")[-1].split(".")[0])

        model.load_state_dict(torch.load(pretrained, map_location=device))

        print("load pretrained model from ", pretrained, " at epoch ", last_epoch)

        start_epoch = last_epoch + 1

    model.to(device)

    model.train()

    # use a leaning rate scheduler
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=300
    )

    lossfn = torch.nn.MSELoss(reduction="mean").to(device)

    writer = SummaryWriter(log_dir=logdir)

    epochs = 500

    best_test_loss = float("inf")

    # Train the model
    for epoch in range(start_epoch, epochs):

        train_loss_value = 0.0

        for x, y, _ in train_loader:

            optimizer.zero_grad()

            y_pred = model(x)

            loss = lossfn(y_pred, y)

            loss.backward()

            optimizer.step()

            train_loss_value += loss.item()

        train_loss_value /= len(train_loader)

        print(f"Epoch {epoch}, Loss: {train_loss_value}")

        scheduler.step()

        if write_log:
            writer.add_scalar("Loss/train", train_loss_value, epoch)

        with torch.no_grad():
            test_loss_value = 0.0

            for x, y, _ in test_loader:

                y_pred = model(x)

                loss = lossfn(y_pred, y)

                test_loss_value += loss.item()

            test_loss_value /= len(test_loader)

            if epoch > save_after_epoch and test_loss_value < best_test_loss:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        checkpoint_dir, f"{model.__class__.__name__}_{epoch}.pth"
                    ),
                )

                best_test_loss = test_loss_value

            print(f"Epoch {epoch}, Test Loss: {test_loss_value}")

            if write_log:
                writer.add_scalar("Loss/test", test_loss_value, epoch)

    writer.close()


if __name__ == "__main__":

    # check env variable `BASE_DIR`
    datadir = os.path.join(os.getenv("BASE_DIR"), "videopose3d_euler_dataset_trunk30")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Datensatz(datadir)

    print(len(dataset))

    feature, target, metadata = dataset[0]

    print(feature.shape, target.shape, metadata)

    # Split the dataset into train and test sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    batch_size = 128

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(len(train_loader), len(test_loader))

    train(train_loader, test_loader, save_after_epoch=50)

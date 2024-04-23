import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv

load_dotenv()

from dataset import Datensatz


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


class MultiHeadAttentionWrapper(nn.Module):
    """
    each attention head in multi-head attention can potentially learn to focus on different parts of the input sequence,
    capturing various aspects or relationships within the data.
    This diversity in representation is key to the success of multi-head attention.

    Multi-head attention can also be more efficient, especially in terms of parallel computation.
    Each head can be processed independently,
    making it well-suited for modern hardware accelerators like GPUs or TPUs that excel at parallel processing.

    eg. the 7B Llama 2 model uses 32 attention heads.
    """

    def __init__(
        self, input_seq_len, output_seq_len, d_in, d_out_kq, d_out_v, num_heads
    ):
        super().__init__()
        # each self-attention head will have its own set of weight matrices, they work in parallel
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v) for _ in range(num_heads)]
        )

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.d_in = d_in
        self.d_out_kq = d_out_kq
        self.d_out_v = d_out_v

        fc1_in = int(input_seq_len * d_out_v)
        fc1_out = int(input_seq_len * d_out_v * 2)
        fc2_out = input_seq_len * 2

        # print(fc1_in, fc1_out, fc2_in, fc2_out)

        self.fc1 = nn.ModuleList([nn.Linear(fc1_in, fc1_out) for _ in range(num_heads)])
        self.activ = nn.Tanh()
        self.fc2 = nn.ModuleList(
            [nn.Linear(fc1_out, fc2_out) for _ in range(num_heads)]
        )

        # target is of shape (12, 3)
        self.fc3 = nn.Linear(fc2_out * num_heads, output_seq_len * d_in * 2)

    def forward(self, x):

        x_linear_heads = []

        for i, head in enumerate(self.heads):
            xi = head(x)
            # flatten for each batch
            xi = xi.view(-1, self.input_seq_len * self.d_out_v)

            xi = self.fc1[i](xi)
            xi = self.activ(xi)
            xi = self.fc2[i](xi)

            x_linear_heads.append(xi)

        # (input_seq_len * num_heads)
        x = torch.cat(x_linear_heads, dim=-1)
        x = self.activ(x)
        x = self.fc3(x)

        return x
        # return x.reshape(-1, self.output_seq_len, self.d_in)


class RNNAttention(nn.Module):
    def __init__(
        self,
        input_seq_len,
        output_seq_len,
        d_in,
        d_out_kq,
        d_out_v,
        num_heads,
        rnn_input_size,
        rnn_hidden_size,
        rnn_num_layers,
        total_sequence_length=30,
    ):
        super().__init__()

        self.attention = MultiHeadAttentionWrapper(
            input_seq_len, output_seq_len, d_in, d_out_kq, d_out_v, num_heads
        )

        self.gru = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(rnn_hidden_size, output_seq_len * d_in)

        self.output_seq_len = output_seq_len
        self.d_in = d_in
        self.total_sequence_length = total_sequence_length

    def forward(self, x):

        # the size of x is (batch_size, sequence_length, bone_length, d_in),
        # where sequence_length=30, bone_length=17
        # pass each item in the batch to the attention model
        seq_attention = torch.stack([self.attention(xi) for xi in x], dim=0)
        # the size of `seq_attention` is (batch_size, sequence_length, output_seq_len*d_in*2)

        # seq_attention to a GRU layer
        rnn_output, _ = self.gru(seq_attention)
        # the size of `rnn_output` is (batch_size, sequence_length, rnn_hidden_size),
        # where sequence_length=30, rnn_hidden_size=output_seq_len*d_in=17*3

        # pass the output of the GRU layer to a fully connected layer
        output = self.fc(rnn_output)

        # print(output.shape)

        return output.reshape(
            -1, self.total_sequence_length, self.output_seq_len, self.d_in
        )


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

    input_seq_len, output_seq_len = 17, 12
    d_in, d_out_kq, d_out_v, num_heads = 3, 6, 8, 4

    rnn_input_size, rnn_hidden_size, rnn_num_layers = (
        output_seq_len * d_in * 2,
        output_seq_len * d_in,
        2,
    )

    model = RNNAttention(
        input_seq_len,
        output_seq_len,
        d_in,
        d_out_kq,
        d_out_v,
        num_heads,
        rnn_input_size,
        rnn_hidden_size,
        rnn_num_layers,
        total_sequence_length=30,
    )

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

    # sys path append ../constants

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

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(len(train_loader), len(test_loader))

    train(train_loader, test_loader, save_after_epoch=50)

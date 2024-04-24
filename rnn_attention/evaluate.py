import os

import torch
from dotenv import load_dotenv

from dataset import Datensatz
from train import RNNAttention

load_dotenv()


if __name__ == "__main__":

    import random

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    checkpoint = os.path.join("checkpoints", "RNNAttention_281.pth")

    model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))

    model.eval()

    datadir = os.path.join(os.getenv("BASE_DIR"), "videopose3d_euler_dataset_trunk30")

    dataset = Datensatz(datadir)

    # take a random sample
    # feature, target, metadata = random.choice(dataset)
    feature, target, metadata = dataset[0]

    target_pred = model(feature)

    torch.set_printoptions(precision=4)

    print("metadata", metadata)

    # print("feature", feature)

    print("target_pred", target_pred[0])
    print("target", target[0])

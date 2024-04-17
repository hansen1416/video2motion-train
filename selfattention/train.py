import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import SelfAttention, MultiHeadAttentionWrapper
from dataset import Datensatz

if __name__ == "__main__":

    import sys

    # sys path append ../constants
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from constants import BASE_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = os.path.join(
        BASE_DIR,
        "video2motion",
        "videopose3d_euler_pose_dataset",
    )

    dataset = Datensatz(dataset_dir)

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

    # print(len(train_loader), len(test_loader))

    input_seq_len, output_seq_len = 17, 12
    d_in, d_out_kq, d_out_v, num_heads = 3, 2, 4, 4

    mha = MultiHeadAttentionWrapper(
        input_seq_len, output_seq_len, d_in, d_out_kq, d_out_v, num_heads
    )

    mha.to(device)

    for i, (x, y, m) in enumerate(train_loader):
        # x, y = x.to(device), y.to(device)

        res = mha(x)

        print(res.size())
        #
        break

    # exit()

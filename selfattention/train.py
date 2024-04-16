import os

import torch
from torch import nn
import torch.nn.functional as F

from model import SelfAttention, MultiHeadAttentionWrapper
from dataset import Datensatz

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = os.path.join(
        "d:\\",
        "video2motion",
        "videopose3d_euler_pose_dataset",
    )

    dataset = Datensatz(dataset_dir)

    print(len(dataset))

    feature, target, metadata = dataset[0]

    print(feature.shape, target.shape, metadata)

    # exit()

    d_in, d_out_kq, d_out_v, num_heads = 3, 2, 4, 4

    mha = MultiHeadAttentionWrapper(d_in, d_out_kq, d_out_v, num_heads)

    mha.to(device)

    res = mha(feature)

    print(res.size())

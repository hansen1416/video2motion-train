import torch
from torch import Tensor
from torch import nn


class MediapipeTransferLinear(nn.Module):
    def __init__(self):
        super(MediapipeTransferLinear, self).__init__()
        self.linear1 = nn.Linear(99, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 66)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.linear3(x)
        x = nn.functional.relu(x)
        x = self.linear4(x)

        # Reshape to (batch_size, 22, 3)
        x = x.reshape(-1, 22, 3)

        return x

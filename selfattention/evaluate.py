import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from train import MultiHeadAttentionWrapper
from dataset import Datensatz

input_seq_len, output_seq_len = 17, 12
d_in, d_out_kq, d_out_v, num_heads = 3, 6, 8, 4

model = MultiHeadAttentionWrapper(
    input_seq_len, output_seq_len, d_in, d_out_kq, d_out_v, num_heads
)

model.load_state_dict(torch.load(os.path.join("checkpoints", "model.pt")))
model.eval()

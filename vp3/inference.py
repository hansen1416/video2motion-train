import os

import numpy as np
import torch

from lstm import Seq2SeqLSTM
from dataset import Datensatz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = Datensatz()

model = Seq2SeqLSTM(input_size=17 * 3, hidden_size=64, num_layers=2)

checkpoint = os.path.join("checkpoints", f"{model.__class__.__name__}_290.pth")

model.load_state_dict(torch.load(checkpoint, map_location=device))

model.eval()

seq_length = 29

with torch.no_grad():

    # genrate random int in the range of ds.seq_length
    # to get a random sequence
    indices = np.random.randint(low=0, high=len(ds), size=10)

    # create empty tensor of shape (len(indices), seq_length, 16, 3)
    outputs = torch.zeros(len(indices), seq_length, 16, 3)

    # print(outputs.shape)

    for i, idx in enumerate(indices):

        input_tensor, _ = ds[idx]

        # wrap input_tensor in a batch
        input_tensor = input_tensor.unsqueeze(0)

        output_tensor = model(input_tensor.view(1, seq_length, -1))

        output_tensor = output_tensor.reshape(seq_length, 16, 3)

        # print(output_tensor.shape)

        outputs[i] = output_tensor

    # print(outputs)
    # print(outputs.shape)

    # save outputs to a .npy file
    np.save("outputs.npy", outputs.to("cpu").numpy())

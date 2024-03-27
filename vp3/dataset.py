import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class Datensatz(Dataset):

    def __init__(self) -> None:

        input_data_dir = os.path.join(
            os.path.expanduser("~"), "Documents", "video2motion", "results3d_dataset"
        )

        self.input_data = np.load(os.path.join(input_data_dir, "res3d_data.npy"))
        # res3d_metadata.pkl
        self.input_metadata = pickle.load(
            open(os.path.join(input_data_dir, "res3d_metadata.pkl"), "rb")
        )

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__():
        pass


if __name__ == "__main__":

    ds = Datensatz()
    print(len(ds))

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
        # res3d_data.npy, 3d joints positions predicted by videopose3d
        self.input_data = np.load(os.path.join(input_data_dir, "res3d_data_trim.npy"))
        # res3d_metadata.pkl, metadata for the 3d joints positions, animation names, start/end frames.
        self.input_metadata = pickle.load(
            open(os.path.join(input_data_dir, "res3d_metadata.pkl"), "rb")
        )
        # anim_euler_data.npy, euler angles of the animations
        self.eulers = np.load(os.path.join(input_data_dir, "anim_euler_data_trim.npy"))

        # these 3 datasets should have the same length
        assert (
            self.input_data.shape[0] == len(self.input_metadata) == self.eulers.shape[0]
        ), "Datasets have different lengths"

    def __len__(self):
        # 17112
        return self.input_data.shape[0]

    def __getitem__(self, idx):

        # ('Angry Gesture (1)-30-0', 120, 150)
        # print(self.input_metadata[idx])

        return (
            torch.tensor(self.input_data[idx], dtype=torch.float32),
            torch.tensor(self.eulers[idx], dtype=torch.float32),
        )


if __name__ == "__main__":

    ds = Datensatz()
    print(len(ds))

    features, targets = ds[100]

    print(features.shape, targets.shape)

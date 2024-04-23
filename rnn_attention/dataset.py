import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from dotenv import load_dotenv

load_dotenv()


class Datensatz(Dataset):

    def __init__(
        self,
        dataset_folder,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.features = torch.tensor(
            np.load(os.path.join(dataset_folder, "features.npy")), dtype=torch.float32
        ).to(self.device)
        self.targets = torch.tensor(
            np.load(os.path.join(dataset_folder, "targets.npy")), dtype=torch.float32
        ).to(self.device)

        with open(os.path.join(dataset_folder, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)

        assert len(self.features) == len(
            self.targets
        ), "The number of rows in the features and targets are not equal."

        assert len(self.features) == len(
            self.metadata
        ), "The number of rows in the features and metadata are not equal."

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.metadata[idx]


if __name__ == "__main__":

    # check env variable `BASE_DIR`
    datadir = os.path.join(os.getenv("BASE_DIR"), "videopose3d_euler_dataset_trunk30")

    # check if the dataset folder exists
    assert os.path.exists(datadir), f"Dataset folder {datadir} does not exist."

    dataset = Datensatz(datadir)

    print(f"Number of samples in the dataset: {len(dataset)}")
    print(dataset[0])

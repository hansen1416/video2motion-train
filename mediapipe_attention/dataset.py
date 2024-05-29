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
        feature_file_path,
        target_file_path,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        features = torch.tensor(np.load(feature_file_path), dtype=torch.float32).to(
            self.device
        )
        targets = torch.tensor(np.load(target_file_path), dtype=torch.float32).to(
            self.device
        )

        # check features, any row contains 0 data more than 30%, remove them
        mask = torch.all(features == 0, dim=1)

        print(mask)

        features = features[~mask]

        self.features = features
        self.targets = targets

        assert len(self.features) == len(
            self.targets
        ), "The number of rows in the features and targets are not equal."

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


if __name__ == "__main__":

    # check env variable `BASE_DIR`
    feature_file_path = os.path.join(
        os.getenv("BASE_DIR"), "mediapipe-euler-dataset", "features.npy"
    )

    target_file_path = os.path.join(
        os.getenv("BASE_DIR"), "mediapipe-euler-dataset", "targets.npy"
    )

    # check if the dataset folder exists
    assert os.path.exists(
        feature_file_path
    ), f"Dataset folder {target_file_path} does not exist."

    dataset = Datensatz(
        feature_file_path=feature_file_path, target_file_path=target_file_path
    )

    print(f"Number of samples in the dataset: {len(dataset)}")

    feature, target = dataset[0]

    print(feature.shape, target.shape)

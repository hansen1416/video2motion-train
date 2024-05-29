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

        features = np.load(feature_file_path)
        targets = np.load(target_file_path)

        mask = []

        # check features, any row contains 0 data more than 30%, remove them
        for i in range(features.shape[0]):
            # Assuming you have your data in a NumPy array named 'data'
            num_zeros = np.count_nonzero(
                features[i] == 0.0
            )  # Count the number of elements equal to 0.0

            if num_zeros / features[i].size > 0.2:
                mask.append(i)

        features = np.delete(features, mask, axis=0)
        targets = np.delete(targets, mask, axis=0)

        self.features = torch.tensor(features, dtype=torch.float32).to(self.device)
        self.targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

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

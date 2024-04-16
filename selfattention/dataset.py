import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class Datensatz(Dataset):

    def __init__(
        self,
        dataset_folder,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(dataset_folder, "features.pkl"), "rb") as f:
            videopose3d = pickle.load(f)

        with open(os.path.join(dataset_folder, "targets.pkl"), "rb") as f:
            anim_eulers = pickle.load(f)

        with open(os.path.join(dataset_folder, "metadata.pkl"), "rb") as f:
            anim_metadata = pickle.load(f)

        assert len(videopose3d) == len(
            anim_eulers
        ), "The number of rows in the features and targets are not equal."

        assert len(videopose3d) == len(
            anim_metadata
        ), "The number of rows in the features and metadata are not equal."

        features = []
        targets = []
        metadata = []

        # i is animation wise
        for i in range(len(videopose3d)):
            # j is frame wise
            for j in range(len(videopose3d[i])):
                features.append(videopose3d[i][j])
                targets.append(anim_eulers[i][j])
                metadata.append(anim_metadata[i].update({"frame": j}))

        features = np.array(features)
        targets = np.array(targets)

        # save it to npy
        np.save(os.path.join(".", "features.npy"), features)
        np.save(os.path.join(".", "targets.npy"), targets)

        with open(os.path.join(".", "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        print(features.shape)
        print(targets.shape)
        print(len(metadata))

    def __len__(self) -> int:
        return sum([len(v) for v in self.features])

    def __getitem__(self, idx):
        return None, None


if __name__ == "__main__":

    dataset_dir = os.path.join(
        os.path.expanduser("~"),
        "Documents",
        "video2motion",
        "videopose3d_euler_dataset",
    )

    dataset = Datensatz(dataset_dir)

    # print(len(dataset))

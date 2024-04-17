import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def flatten_anim_data():
    """
    from videopose3d_euler_dataset to videopose3d_euler_pose_dataset

    from animation data to frame wise data

    """

    dataset_dir = os.path.join(
        BASE_DIR,
        "video2motion",
        "videopose3d_euler_dataset",
    )

    with open(os.path.join(dataset_dir, "features.pkl"), "rb") as f:
        features = pickle.load(f)

    with open(os.path.join(dataset_dir, "targets.pkl"), "rb") as f:
        targets = pickle.load(f)

    with open(os.path.join(dataset_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    merkmale = []
    ziele = []
    metadaten = []

    for i in range(len(features)):
        for j in range(len(features[i])):

            pose = np.array(features[i][j])
            euler = np.array(targets[i][j])
            meta = {
                "name": metadata[i]["name"],
                "total_frame": metadata[i]["total_frame"],
                "frame": j,
            }

            # print(pose.shape, euler.shape, meta)

            merkmale.append(pose)
            ziele.append(euler)
            metadaten.append(meta)

        #     break
        # break

    merkmale = np.array(merkmale)
    ziele = np.array(ziele)

    print(merkmale.shape, ziele.shape, len(metadaten))

    np.save(
        os.path.join(
            BASE_DIR,
            "video2motion",
            "videopose3d_euler_pose_dataset",
            "features1.npy",
        ),
        merkmale,
    )

    np.save(
        os.path.join(
            BASE_DIR,
            "video2motion",
            "videopose3d_euler_pose_dataset",
            "targets1.npy",
        ),
        ziele,
    )

    with open(
        os.path.join(
            BASE_DIR,
            "video2motion",
            "videopose3d_euler_pose_dataset",
            "metadata1.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(metadaten, f)


class Datensatz(Dataset):

    def __init__(
        self,
        dataset_folder,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # numpy to tensor
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

    import sys

    # sys path append ../constants
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from constants import BASE_DIR

    dataset_dir = os.path.join(
        BASE_DIR,
        "video2motion",
        "videopose3d_euler_pose_dataset",
    )

    dataset = Datensatz(dataset_dir)

    print(len(dataset))

    feature, target, metadata = dataset[0]

    print(feature.shape, target.shape, metadata)

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MDataset(Dataset):

    def __init__(
        self,
        mediapipe_joined_dir,
        anim_euler_joined_dir,
    ) -> None:
        super().__init__()

        self.mediapipe_joined_dir = mediapipe_joined_dir
        self.anim_euler_joined_dir = anim_euler_joined_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mediapipe_joined = np.load(
            os.path.join(self.mediapipe_joined_dir, "joined.npy")
        )

        self.anim_euler_joined = np.load(
            os.path.join(self.anim_euler_joined_dir, "joined.npy")
        )

        assert (
            self.mediapipe_joined.shape[0] == self.anim_euler_joined.shape[0]
        ), "The number of samples in the datasets are not equal."

    def __len__(self) -> int:
        return self.anim_euler_joined.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.mediapipe_joined[idx], dtype=torch.float32).to(
                self.device
            ),
            torch.tensor(self.anim_euler_joined[idx], dtype=torch.float32).to(
                self.device
            ),
        )


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    from constants import MEDIAPIPE_JOINED_DIR, ANIM_EULER_JOINED_DIR

    ds = MDataset(MEDIAPIPE_JOINED_DIR, ANIM_EULER_JOINED_DIR)

    print(len(ds))

    mediapipe_input, anim_euler_output = ds[0]

    print(mediapipe_input.shape, anim_euler_output.shape)

    dl = DataLoader(ds, batch_size=32, shuffle=True)

    print(len(dl))

    for i, (mediapipe, anim_euler) in enumerate(dl):
        pass

    print(i)

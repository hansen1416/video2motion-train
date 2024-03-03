import os
from typing import Tuple, Dict, List, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class MediapipeDataset(Dataset):

    def __init__(
        self,
        inputs_dir: str,
        outputs_dir: str,
        indices_to_file_index: Dict,
        data_indices_in_files: List[List[int]],
        device="cpu",
    ) -> None:
        """
        Args:
            inputs_dir (str): The directory containing the input data.
            outputs_dir (str): The directory containing the output data.
            indices_to_file_index (dict): A dictionary mapping the indices of the data to the file indices.
            data_indices_in_files (list): A list of lists containing the sizes of the data in each file.
        """

        super().__init__()

        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.indices_to_file_index = indices_to_file_index
        self.data_indices_in_files = data_indices_in_files
        self.device = device

        # current file data
        self.current_file_idx: int = None
        self.current_inputs: torch.Tensor = None
        self.current_outputs: torch.Tensor = None

    def __len__(self) -> int:
        return len(self.indices_to_file_index)

    def load_file(self, file_idx) -> None:
        """
        load the data of the file which contains the data of the given index into memory.
        """

        self.current_file_idx = file_idx
        self.current_inputs = np.load(
            os.path.join(self.inputs_dir, f"inputs_{file_idx}.npy")
        )
        self.current_outputs = np.load(
            os.path.join(self.outputs_dir, f"outputs_{file_idx}.npy")
        )

        self.current_inputs = torch.tensor(self.current_inputs, dtype=torch.float32).to(
            self.device
        )

        self.current_outputs = torch.tensor(
            self.current_outputs, dtype=torch.float32
        ).to(self.device)

        # print(f"loaded file {file_idx}")

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input and output data of the given index.
        if the data is not in memory, load the file containing the data into memory.
        """

        file_idx = self.indices_to_file_index[idx]

        if self.current_file_idx != file_idx:
            self.load_file(file_idx)

        # get the index of the data in the current file, minus the sum of the sizes of the previous files
        idx -= sum(
            [
                len(sizes)
                for i, sizes in enumerate(self.data_indices_in_files)
                if i < file_idx
            ]
        )

        return self.current_inputs[idx], self.current_outputs[idx]


class FilewiseShuffleSampler(Sampler):
    """
    Shuffle the indices of the data in a filewise manner.
    This is useful when the data is stored in multiple files and you want to shuffle the data within each file.
    """

    def __init__(
        self, data_indices_in_files: Sequence[Sequence[int]] = None, generator=None
    ) -> None:
        """
        Args:
            data_indices_in_files (list): A list of list of indeices for each file.
            generator (torch.Generator, optional): Generator used for shuffling. Default is None.
        """
        self.data_indices_in_files = data_indices_in_files
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for data_indices in self.data_indices_in_files:
            for i in torch.randperm(len(data_indices), generator=self.generator):
                yield data_indices[i]

    def __len__(self) -> int:
        return sum([len(sizes) for sizes in self.data_indices_in_files])


def fetch_datatset_info(inputs_dir, outputs_dir) -> Tuple[Dict, List]:

    file_idx = 0
    indices_to_file_index = {}
    accumulated_count = 0
    data_indices_in_files = []

    for f in os.listdir(inputs_dir):
        data = np.load(os.path.join(inputs_dir, f))

        indices_to_file_index.update(
            {i + accumulated_count: file_idx for i in range(data.shape[0])}
        )

        data_indices_in_files.append(
            list(range(accumulated_count, accumulated_count + data.shape[0]))
        )

        file_idx += 1
        accumulated_count += data.shape[0]

    return indices_to_file_index, data_indices_in_files


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from FilewiseShuffleSampler import FilewiseShuffleSampler

    inputs_dir = os.path.join(os.path.dirname(__file__), "data", "inputs")
    outputs_dir = os.path.join(os.path.dirname(__file__), "data", "outputs")

    if not os.path.exists(inputs_dir):
        os.makedirs(inputs_dir)

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    indices_to_file_index, data_indices_in_files = fetch_datatset_info(
        inputs_dir, outputs_dir
    )

    print(f"indices_to_file_index size: {len(indices_to_file_index)}")
    print(f"data_indices_in_files: {data_indices_in_files}")

    dataset = MediapipeDataset(
        inputs_dir,
        outputs_dir,
        indices_to_file_index,
        data_indices_in_files,
        device="cpu",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        sampler=FilewiseShuffleSampler(data_indices_in_files=data_indices_in_files),
    )

    for inputs, outputs in dataloader:
        print(inputs.shape, outputs.shape)
        # break

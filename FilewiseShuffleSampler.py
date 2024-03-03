import torch
from torch.utils.data import Sampler
from typing import Iterator, Sequence


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


if __name__ == "__main__":

    data_indices_in_files = [
        list(range(10, 21)),
        list(range(30, 41)),
        list(range(51, 60)),
        list(range(70, 73)),
    ]
    sampler = FilewiseShuffleSampler(data_indices_in_files)

    for i in sampler:
        # Your data loading logic using the index 'i'
        # pass
        print(i)
        pass

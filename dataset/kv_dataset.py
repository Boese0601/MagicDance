""" KV Dataset from Peter's branch """
import random
import json
import io
import os
from base64 import b64decode
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from typing import Any, List, Callable, Union
from .hdfs_io import hlist_files

def partition_by_size(data: List[Any], size: int) -> List[List[Any]]:
    """
    Partition a list by size.
    When indivisible, the last group contains fewer items than the target size.

    Examples:
        - data: [1,2,3,4,5]
        - size: 2
        - return: [[1,2], [3,4], [5]]
    """
    return [data[i:i+size] for i in range(0, len(data), size)]


def partition_by_groups(data: List[Any], groups: int) -> List[List[Any]]:
    """
    Partition a list by groups.
    When indivisible, some groups may have more items than others.

    Examples:
        - data: [1,2,3,4,5]
        - groups: 2
        - return: [[1,3,5], [2,4]]
    """
    return [data[i::groups] for i in range(groups)]


class KVDataset(IterableDataset):
    """
    A base iterable dataset for loading data using KVReader. Iterator returns raw data value.
    Arguments:
        paths: a list of directory path that contains *.index files.
        rank: current distributed process rank. Used for partitioning the dataset.
        world_size: total distributed processes. Used for partitioning the dataset.
        shuffle: whether to shuffle the dataset order.
    """
    def __init__(
        self,
        paths: List[str],
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        repeat: bool = True,
        reader_threads: int = 8,
        chunk_size: int = 100,
    ):
        super().__init__()
        assert len(paths) > 0, "Paths must not be empty."
        assert rank < world_size and rank >= 0, "Rank must be >= 0 and < world_size."
        assert world_size >= 1, "World_size must be >= 1."
        self.paths = paths
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.reader_threads = reader_threads
        self.chunk_size = chunk_size
        # List .index files under the path directory
        self.filepaths = []
        for filepath in hlist_files(self.paths):
            filename, extension = os.path.splitext(filepath)
            if extension == ".index":
                self.filepaths.append(filename)
        # assert len(self.filepaths) > 0, "Could not find any .index files under paths."
        
        # Partition the list by rank and world_size
        self.filepaths = partition_by_groups(self.filepaths, world_size)[rank]
        
        # Shuffle if needed.
        if self.shuffle:
            random.shuffle(self.filepaths)

    def __iter__(self):
        filepaths = self.filepaths
        # Partition the list by dataloader worker if needed.
        worker_info = get_worker_info()
        while True:
            # Loop through all the .index files
            for filepath in filepaths:
                try:
                    reader = KVReader(filepath, self.reader_threads)
                    data_source = "/".join(filepath.rsplit('/', 2)[-2:])
                    keys = reader.list_keys()
                    # print(keys)
                    if worker_info is not None:
                        keys = partition_by_groups(sorted(keys), worker_info.num_workers)[worker_info.id]
                    if self.shuffle:
                        random.shuffle(keys)
                    # Load 100 items at a time.
                    for key_batch in partition_by_size(keys, self.chunk_size):
                        value_batch = reader.read_many(key_batch)
                        for value in value_batch:
                            yield value
                except Exception as ex:
                    print(f"KVDataset got unexpected exception: {ex}")
                    continue
            if not self.repeat:
                break
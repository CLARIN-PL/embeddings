from abc import ABC
from enum import Enum
from typing import Any, Dict, Generic, Sequence, TypeVar, Union

from torch.utils.data import DataLoader

Data = TypeVar("Data")
LightingDataLoaders = Union[
    DataLoader[Any],
    Sequence[DataLoader[Any]],
    Sequence[Sequence[DataLoader[Any]]],
    Sequence[Dict[str, DataLoader[Any]]],
    Dict[str, DataLoader[Any]],
    Dict[str, Dict[str, DataLoader[Any]]],
    Dict[str, Sequence[DataLoader[Any]]],
]


class BaseDataset(ABC, Generic[Data]):
    def __repr__(self) -> str:
        return type(self).__name__


class Dataset(BaseDataset[str]):
    def __init__(self, dataset: str, **load_dataset_kwargs: Any):
        super().__init__()
        self.dataset = dataset
        self.load_dataset_kwargs = load_dataset_kwargs


class LightingDataModuleSubset(str, Enum):
    TRAIN = "train"
    VALIDATION = "dev"
    TEST = "test"
    PREDICT = "predict"

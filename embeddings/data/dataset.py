from abc import ABC
from pathlib import Path
from typing import Any, Generic, TypeVar, Union

Data = TypeVar("Data")


class Dataset(ABC, Generic[Data]):
    def __repr__(self) -> str:
        return type(self).__name__


class LocalDataset(Dataset[str]):
    def __init__(self, dataset: Union[str, Path], **load_dataset_kwargs: Any):
        super().__init__()
        self.dataset = dataset
        self.load_dataset_kwargs = load_dataset_kwargs


class HuggingFaceDataset(Dataset[str]):
    def __init__(self, dataset: Union[str, Path], **load_dataset_kwargs: Any):
        super().__init__()
        self.dataset = dataset
        self.load_dataset_kwargs = load_dataset_kwargs

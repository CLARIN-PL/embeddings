import abc
from abc import ABC
from os.path import exists, isdir
from typing import Generic, TypeVar, Union

import datasets

from embeddings.data.dataset import BaseDataset, Dataset

Input = TypeVar("Input")
Output = TypeVar("Output")


class DataLoader(ABC, Generic[Input, Output]):
    @abc.abstractmethod
    def load(self, dataset: BaseDataset[Input]) -> Output:
        pass


class HuggingFaceDataLoader(DataLoader[str, datasets.DatasetDict]):
    def load(self, dataset: BaseDataset[str]) -> datasets.DatasetDict:
        assert isinstance(dataset, Dataset)
        result = datasets.load_dataset(dataset.dataset, **dataset.load_dataset_kwargs)
        assert isinstance(result, datasets.DatasetDict)
        return result


class HuggingFaceLocalDataLoader(DataLoader[str, datasets.DatasetDict]):
    def load(self, dataset: BaseDataset[str]) -> datasets.DatasetDict:
        assert isinstance(dataset, Dataset)
        result = datasets.load_from_disk(dataset.dataset)
        assert isinstance(result, datasets.DatasetDict)
        return result


HF_DATALOADERS = Union[HuggingFaceDataLoader, HuggingFaceLocalDataLoader]


def get_hf_dataloader(dataset: Dataset) -> HF_DATALOADERS:
    if exists(dataset.dataset):
        if not isdir(dataset.dataset):
            raise NotImplementedError(
                "Reading from file is currently not supported. "
                "Pass dataset directory or HuggingFace repository name"
            )
        return HuggingFaceLocalDataLoader()
    return HuggingFaceDataLoader()

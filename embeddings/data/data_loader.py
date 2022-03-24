import abc
import pickle
from abc import ABC
from os.path import exists, isdir
from pathlib import Path
from typing import Generic, TypeVar, Union

import datasets
from flair.data import Corpus
from flair.datasets import ColumnDataset

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


class PickleFlairCorpusDataLoader(DataLoader[str, Corpus]):
    def load(self, dataset: BaseDataset[str]) -> Corpus:
        assert isinstance(dataset, Dataset)
        with open(dataset.dataset, "rb") as file:
            corpus = pickle.load(file)
        assert isinstance(corpus, Corpus)
        return corpus


class ConllFlairCorpusDataLoader(DataLoader[str, Corpus]):
    DEFAULT_COLUMN_FORMAT = {0: "text", 1: "tag"}
    DEFAULT_FLAIR_SUBSET_NAMES = ["train", "dev", "test"]

    def load(self, dataset: BaseDataset[str]) -> Corpus:
        assert isinstance(dataset, Dataset)
        assert exists(dataset.dataset)
        flair_datasets = {}
        dataset_dir = Path(dataset.dataset)
        for subset_name in self.DEFAULT_FLAIR_SUBSET_NAMES:
            output_path = dataset_dir.joinpath(f"{subset_name}.tsv")
            if output_path.exists():
                flair_datasets[subset_name] = ColumnDataset(output_path, self.DEFAULT_COLUMN_FORMAT)
            else:
                flair_datasets[subset_name] = None

        return Corpus(
            train=flair_datasets["train"],
            dev=flair_datasets["dev"],
            test=flair_datasets["test"],
            sample_missing_splits=False,
        )


HF_DATALOADERS = Union[HuggingFaceDataLoader, HuggingFaceLocalDataLoader]
FLAIR_DATALOADERS = Union[
    HuggingFaceDataLoader, PickleFlairCorpusDataLoader, ConllFlairCorpusDataLoader
]


def get_hf_dataloader(dataset: Dataset) -> HF_DATALOADERS:
    if exists(dataset.dataset):
        if not isdir(dataset.dataset):
            raise NotImplementedError(
                "Reading from file is currently not supported. "
                "Pass dataset directory or HuggingFace repository name"
            )
        return HuggingFaceLocalDataLoader()
    return HuggingFaceDataLoader()


def get_flair_dataloader(dataset: Dataset) -> FLAIR_DATALOADERS:
    if exists(dataset.dataset):
        if isdir(dataset.dataset):
            return ConllFlairCorpusDataLoader()
        else:
            return PickleFlairCorpusDataLoader()
    return HuggingFaceDataLoader()

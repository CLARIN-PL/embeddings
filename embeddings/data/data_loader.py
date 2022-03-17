import abc
import pickle
from abc import ABC
from os.path import exists, isdir
from pathlib import Path
from typing import Generic, TypeVar, Union

import datasets
from flair.data import Corpus
from flair.datasets import ColumnDataset

from embeddings.data.dataset import Dataset, LoadableDataset

Input = TypeVar("Input")
Output = TypeVar("Output")


class DataLoader(ABC, Generic[Input, Output]):
    @abc.abstractmethod
    def load(self, dataset: Dataset[Input]) -> Output:
        pass


class HuggingFaceDataLoader(DataLoader[str, datasets.DatasetDict]):
    def load(self, dataset: Dataset[str]) -> datasets.DatasetDict:
        if isinstance(dataset, LoadableDataset):
            result = datasets.load_dataset(dataset.dataset, **dataset.load_dataset_kwargs)
            assert isinstance(result, datasets.DatasetDict)
            return result
        else:
            raise ValueError("This DataLoader should be used with HuggingFaceDataset only.")


class HuggingFaceLocalDataLoader(DataLoader[str, datasets.DatasetDict]):
    def load(self, dataset: Dataset[str]) -> datasets.DatasetDict:
        if isinstance(dataset, LoadableDataset):
            result = datasets.load_from_disk(str(dataset.dataset))
            assert isinstance(result, datasets.DatasetDict)
            return result
        else:
            raise ValueError("This DataLoader should be used with HuggingFaceDataset only.")


class PickleFlairCorpusDataLoader(DataLoader[str, Corpus]):
    def load(self, dataset: Dataset[str]) -> Corpus:
        assert isinstance(dataset, LoadableDataset)
        with open(dataset.dataset, "rb") as file:
            corpus = pickle.load(file)
        assert isinstance(corpus, Corpus)
        return corpus


class ConllFlairCorpusDataLoader(DataLoader[str, Corpus]):
    DEFAULT_COLUMN_FORMAT = {0: "text", 1: "tag"}
    DEFAULT_FLAIR_SUBSET_NAMES = ["train", "dev", "test"]

    def load(self, dataset: Dataset[str]) -> Corpus:
        assert isinstance(dataset, LoadableDataset)
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


def get_hf_dataloader(dataset: LoadableDataset) -> HF_DATALOADERS:
    if exists(dataset.dataset):
        if not isdir(dataset.dataset):
            raise NotImplementedError(
                "Reading from file is currently not supported. "
                "Pass dataset directory or HuggingFace repository name"
            )
        return HuggingFaceLocalDataLoader()
    return HuggingFaceDataLoader()


def get_flair_dataloader(dataset: LoadableDataset) -> FLAIR_DATALOADERS:
    if exists(dataset.dataset):
        if isdir(dataset.dataset):
            return ConllFlairCorpusDataLoader()
        else:
            return PickleFlairCorpusDataLoader()
    return HuggingFaceDataLoader()

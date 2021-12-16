import abc
import pickle
from abc import ABC
from pathlib import Path
from typing import Generic, TypeVar

import datasets
from flair.data import Corpus
from flair.datasets import ColumnDataset

from embeddings.data.dataset import Dataset, HuggingFaceDataset, LocalDataset

Input = TypeVar("Input")
Output = TypeVar("Output")


class DataLoader(ABC, Generic[Input, Output]):
    @abc.abstractmethod
    def load(self, dataset: Dataset[Input]) -> Output:
        pass


class HuggingFaceDataLoader(DataLoader[str, datasets.DatasetDict]):
    def load(self, dataset: Dataset[str]) -> datasets.DatasetDict:
        if isinstance(dataset, HuggingFaceDataset):
            result = datasets.load_dataset(str(dataset.dataset), **dataset.load_dataset_kwargs)
            assert isinstance(result, datasets.DatasetDict)
            return result
        else:
            raise ValueError("This DataLoader should be used with HuggingFaceDataset only.")


class PickleFlairCorpusDataLoader(DataLoader[str, Corpus]):
    def load(self, dataset: Dataset[str]) -> Corpus:
        assert isinstance(dataset, LocalDataset)
        with open(dataset.dataset, "rb") as file:
            corpus = pickle.load(file)
        assert isinstance(corpus, Corpus)
        return corpus


class ConllFlairCorpusDataLoader(DataLoader[str, Corpus]):
    DEFAULT_COLUMN_FORMAT = {0: "text", 1: "tag"}
    DEFAULT_FLAIR_SUBSET_NAMES = ["train", "dev", "test"]

    def load(self, dataset: Dataset[str]) -> Corpus:
        assert isinstance(dataset, LocalDataset)
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

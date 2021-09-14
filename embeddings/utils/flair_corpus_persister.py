import abc
import pickle
from abc import ABC
from pathlib import Path
from typing import Generic, TypeVar, Union

from flair.datasets import ColumnDataset

from embeddings.utils.loggers import get_logger

Input = TypeVar("Input")
ClassificationCorpus = TypeVar("ClassificationCorpus")
PairClassificationCorpus = TypeVar("PairClassificationCorpus")
SequenceLabelingCorpus = TypeVar("SequenceLabelingCorpus")

_logger = get_logger(__name__)


class FlairCorpusPersister(ABC, Generic[Input]):
    @abc.abstractmethod
    def persist(self, data: Input) -> None:
        pass


class FlairConllPersister(FlairCorpusPersister[SequenceLabelingCorpus]):
    def __init__(self, path: str):
        self.path = Path(path)

    def persist(self, data: SequenceLabelingCorpus) -> None:
        subset_names = ["train", "dev", "test"]
        for subset_name in subset_names:
            try:
                data_subset = getattr(data, subset_name)
                output_path = self.path.joinpath(f"{subset_name}.tsv")
                self._save_conll(data_subset, output_path)
            except AttributeError:
                _logger.info(
                    f"Exception occured when persisting data: {subset_name} dataset not found in the Corpus"
                )
            except TypeError:
                _logger.info(
                    f"Exception occured when persisting data: {subset_name} dataset is None"
                )

    @staticmethod
    def _save_conll(data: ColumnDataset, filepath: Path) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for sentence in data:
                for token in sentence:
                    f.write(f"{token.text}\t{token.get_tag('tag').value}\n")
                f.write("\n")


class FlairPicklePersister(
    FlairCorpusPersister[Union[ClassificationCorpus, PairClassificationCorpus]]
):
    def __init__(self, path: str):
        self.path = path

    def persist(self, data: Union[ClassificationCorpus, PairClassificationCorpus]) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(obj=data, file=f)

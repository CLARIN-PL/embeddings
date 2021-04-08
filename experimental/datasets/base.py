import abc
from abc import ABC

from flair.datasets import ColumnCorpus


class BaseDataset(ABC):
    @abc.abstractmethod
    def to_flair_column_corpus(self) -> ColumnCorpus:
        pass

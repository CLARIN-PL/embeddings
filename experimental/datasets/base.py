import abc

from flair.datasets import ColumnCorpus


class Dataset:
    @abc.abstractmethod
    def to_flair_column_corpus(self) -> ColumnCorpus:
        pass

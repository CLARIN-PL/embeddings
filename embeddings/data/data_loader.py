import abc
from typing import TypeVar, Generic
from abc import ABC
from embeddings.data.dataset import Dataset
from embeddings.utils.loggers import get_logger

Input = TypeVar("Input")
Output = TypeVar("Output")


class DataLoader(ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        self._logger = get_logger(str(self))

    def __repr__(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def load(self, dataset: Dataset[Input]) -> Output:
        pass

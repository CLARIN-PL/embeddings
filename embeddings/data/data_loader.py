import abc
from abc import ABC
from typing import Generic, TypeVar

from embeddings.data.dataset import Dataset

Input = TypeVar("Input")
Output = TypeVar("Output")


class DataLoader(ABC, Generic[Input, Output]):
    @abc.abstractmethod
    def load(self, dataset: Dataset[Input]) -> Output:
        pass

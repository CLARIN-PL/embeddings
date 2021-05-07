import abc
from typing import TypeVar, Generic
from abc import ABC
from embeddings.data.Dataset import Dataset

Input = TypeVar("Input")
Output = TypeVar("Output")


class DataLoader(ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def load(self, dataset: Dataset[Input]) -> Output:
        pass

import abc
from abc import ABC
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


class Embedding(ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def embed(self, data: Input) -> Output:
        pass

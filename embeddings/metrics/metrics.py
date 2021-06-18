import abc
from typing import TypeVar, Generic
from abc import ABC

Input = TypeVar("Input")
Output = TypeVar("Output")


class Metric(ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def compute(self, predictions: Input, references: Input) -> Output:
        pass

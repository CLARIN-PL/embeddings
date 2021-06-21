import abc
from typing import TypeVar, Generic, Any
from abc import ABC

Input = TypeVar("Input")
Output = TypeVar("Output")


class Metric(ABC, Generic[Input, Output]):
    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return self._name

    @abc.abstractmethod
    def compute(self, y_pred: Input, y_true: Input, **kwargs: Any) -> Output:
        pass

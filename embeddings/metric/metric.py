import abc
from abc import ABC
from typing import Any, Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


class Metric(ABC, Generic[Input, Output]):
    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return self._name

    @abc.abstractmethod
    def compute(self, y_true: Input, y_pred: Input, **kwargs: Any) -> Output:
        pass

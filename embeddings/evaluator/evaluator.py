import abc
from abc import ABC
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


class Evaluator(ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def evaluate(self, data: Input) -> Output:
        pass

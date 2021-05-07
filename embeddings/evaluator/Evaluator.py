import abc
from typing import TypeVar, Generic
from abc import ABC

Input = TypeVar("Input")
Output = TypeVar("Output")


class Evaluator(ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def evaluate(self, data: Input) -> Output:
        pass

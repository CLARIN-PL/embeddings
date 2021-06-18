import abc
from typing import TypeVar, Generic
from abc import ABC

Input = TypeVar("Input")
Output = TypeVar("Output")


class Task(ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def task(self, data: Input) -> Output:
        pass

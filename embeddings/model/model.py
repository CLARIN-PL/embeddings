import abc
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


class Model(abc.ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def model(self, data: Input) -> Output:
        pass

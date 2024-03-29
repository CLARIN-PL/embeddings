import abc
from typing import Any, Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


class Model(abc.ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def execute(self, data: Input, **kwargs: Any) -> Output:
        pass

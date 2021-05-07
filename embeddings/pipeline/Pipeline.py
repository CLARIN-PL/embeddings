import abc
from typing import TypeVar, Generic
from abc import ABC

Output = TypeVar("Output")


class Pipeline(ABC, Generic[Output]):
    @abc.abstractmethod
    def run(self) -> Output:
        pass

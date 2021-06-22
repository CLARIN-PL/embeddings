import abc
from abc import ABC
from typing import Generic, TypeVar

Output = TypeVar("Output")


class Pipeline(ABC, Generic[Output]):
    @abc.abstractmethod
    def run(self) -> Output:
        pass

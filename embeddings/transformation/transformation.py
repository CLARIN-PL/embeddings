import abc
from typing import TypeVar, Generic, List, Any
from abc import ABC

from embeddings.utils.loggers import get_logger

Input = TypeVar("Input")
Output = TypeVar("Output")


class Transformation(ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        self._logger = get_logger(str(self))

    def __repr__(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def transform(self, data: Input) -> Output:
        pass


class Transformations(Transformation[Input, Output]):
    def __init__(self, transformations: List[Transformation[Input, Output]]):
        self.transformations = transformations

    def transform(self, data: Input) -> Any:
        for transformation in self.transformations:
            data = transformation.transform(data)

        return data

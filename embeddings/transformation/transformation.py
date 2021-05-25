import abc
from typing import TypeVar, Generic, List, Any
from abc import ABC
from embeddings.utils.loggers import get_logger

Input = TypeVar("Input")
Output = TypeVar("Output")
NewOutput = TypeVar("NewOutput")
OutputInternal = TypeVar("OutputInternal")


class Transformation(ABC, Generic[Input, Output]):
    def __init__(self) -> None:
        self._logger = get_logger(str(self))

    def __repr__(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def transform(self, data: Input) -> Output:
        pass

    def then(
        self, right: "Transformation[Output, NewOutput]"
    ) -> "Transformation[Input, NewOutput]":
        return CombainedTransformations(self, right)


class CombainedTransformations(
    Transformation[Input, Output], Generic[Input, Output, OutputInternal]
):
    def __init__(
        self,
        left: Transformation[Input, OutputInternal],
        right: Transformation[OutputInternal, Output],
    ) -> None:
        self.left = left
        self.right = right

    def transform(self, data: Input) -> Output:
        intermidiate = self.left.transform(data)
        return self.right.transform(intermidiate)

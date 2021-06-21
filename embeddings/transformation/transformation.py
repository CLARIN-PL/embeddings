import abc
from abc import ABC
from typing import Generic, TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")
NewOutput = TypeVar("NewOutput")
OutputInternal = TypeVar("OutputInternal")


class Transformation(ABC, Generic[Input, Output]):
    @abc.abstractmethod
    def transform(self, data: Input) -> Output:
        pass

    def then(
        self, right: "Transformation[Output, NewOutput]"
    ) -> "Transformation[Input, NewOutput]":
        return CombinedTransformations(self, right)


class CombinedTransformations(
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
        intermediate = self.left.transform(data)
        return self.right.transform(intermediate)

import abc
from abc import ABC
from typing import Generic, TypeVar

from embeddings.utils.results_persister import ResultsPersister

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

    def persisting(self, persister: ResultsPersister[Output]) -> "Transformation[Input, Output]":
        return PersistingTransformation(self, persister)


class DummyTransformation(Transformation[Input, Input]):
    def transform(self, data: Input) -> Input:
        return data


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


class PersistingTransformation(Transformation[Input, Output]):
    def __init__(
        self,
        base_transformation: Transformation[Input, Output],
        persister: ResultsPersister[Output],
    ):
        self.base_transformation = base_transformation
        self.persister = persister

    def transform(self, data: Input) -> Output:
        result = self.base_transformation.transform(data)
        self.persister.persist(result)
        return result

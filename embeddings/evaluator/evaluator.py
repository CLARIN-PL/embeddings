import abc
from abc import ABC
from typing import Generic, TypeVar

from embeddings.utils.results_persister import ResultsPersister

Input = TypeVar("Input")
Output = TypeVar("Output")


class Evaluator(ABC, Generic[Input, Output]):
    @abc.abstractmethod
    def evaluate(self, data: Input) -> Output:
        pass

    def persisting(self, persister: ResultsPersister[Output]) -> "Evaluator[Input, Output]":
        return PersistingEvaluator(self, persister)


class PersistingEvaluator(Evaluator[Input, Output]):
    def __init__(
        self, base_evaluator: Evaluator[Input, Output], persister: ResultsPersister[Output]
    ):
        self.base_evaluator = base_evaluator
        self.persister = persister

    def evaluate(self, data: Input) -> Output:
        result = self.base_evaluator.evaluate(data)
        self.persister.persist(result)
        return result

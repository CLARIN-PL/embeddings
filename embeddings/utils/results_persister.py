import abc
from abc import ABC
from typing import Any, Generic, TypeVar

Input = TypeVar("Input")


class ResultsPersister(ABC, Generic[Input]):
    @abc.abstractmethod
    def persist(self, result: Input, **kwargs: Any) -> None:
        pass

from abc import ABC
from typing import Generic, TypeVar

Data = TypeVar("Data")


class Dataset(ABC, Generic[Data]):
    def __repr__(self) -> str:
        return type(self).__name__

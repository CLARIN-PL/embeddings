from typing import TypeVar, Generic
from abc import ABC

Data = TypeVar("Data")


class Dataset(ABC, Generic[Data]):
    def __repr__(self) -> str:
        return type(self).__name__

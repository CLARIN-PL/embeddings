from typing import TypeVar, Generic
from abc import ABC

from embeddings.utils.loggers import get_logger

Data = TypeVar("Data")


class Dataset(ABC, Generic[Data]):
    def __init__(self) -> None:
        self._logger = get_logger(str(self))

    def __repr__(self) -> str:
        return type(self).__name__

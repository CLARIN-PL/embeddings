from typing import TypeVar, Generic
from abc import ABC

Data = TypeVar("Data")


class Dataset(ABC, Generic[Data]):
    def __init__(self) -> None:
        pass

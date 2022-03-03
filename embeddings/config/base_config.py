import abc
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Set

from embeddings.data.io import T_path
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class Config(ABC):
    @classmethod
    @abc.abstractmethod
    def from_yaml(cls, path: T_path) -> "Config":
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        pass


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class BasicConfig(Config, ABC):
    @abc.abstractmethod
    def __post_init__(self) -> None:
        pass

    def _parse_fields(self, keys: Set[str]) -> Dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in keys}


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class AdvancedConfig(Config, ABC):
    @abc.abstractmethod
    def __post_init__(self) -> None:
        pass

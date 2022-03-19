import abc
from dataclasses import dataclass
from typing import Any, Dict, Set, Tuple

from embeddings.data.io import T_path
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class Config(abc.ABC):
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
class BasicConfig(Config, abc.ABC):
    @abc.abstractmethod
    def __post_init__(self) -> None:
        pass

    def _parse_fields(self, keys: Set[str]) -> Dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in keys}

    def _map_parse_fields(self, key_tuples: Set[Tuple[str, str]]):
        return {field_name: getattr(self, attr_name) for attr_name, field_name in key_tuples}

    @classmethod
    def get_config_keys(cls) -> Set[str]:
        config_keys = {key for key in cls.__annotations__.keys() if not key.endswith("_kwargs")}
        return config_keys


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class AdvancedConfig(Config, abc.ABC):
    @abc.abstractmethod
    def __post_init__(self) -> None:
        pass

    @staticmethod
    def from_basic() -> "AdvancedConfig":
        pass

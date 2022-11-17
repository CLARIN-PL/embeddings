import abc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set, Tuple

from embeddings.data.io import T_path
from embeddings.utils.loggers import get_logger
from embeddings.utils.utils import read_yaml

_logger = get_logger(__name__)


@dataclass
class Config(abc.ABC):
    @classmethod
    def from_yaml(cls, path: T_path) -> "Config":
        config = read_yaml(path)
        return cls(**config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Config":
        return cls(**config)


@dataclass
class BasicConfig(Config, abc.ABC):
    def _parse_fields(self, keys: Iterable[str]) -> Dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in keys}

    def _map_parse_fields(self, key_tuples: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
        return {field_name: getattr(self, attr_name) for attr_name, field_name in key_tuples}

    @classmethod
    def get_config_keys(cls) -> Set[str]:
        config_keys = {key for key in cls.__annotations__.keys() if not key.endswith("_kwargs")}
        return config_keys


@dataclass
class AdvancedConfig(Config, abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def from_basic() -> "AdvancedConfig":
        pass

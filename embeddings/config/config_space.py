import abc
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from embeddings.data.io import T_path
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


class ConfigSpace(ABC):
    @classmethod
    @abc.abstractmethod
    def from_yaml(cls, path: T_path) -> "ConfigSpace":
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfigSpace":
        pass

    @staticmethod
    def _check_unmapped_parameters(parameters: Dict[str, Any]) -> None:
        if len(parameters):
            raise ValueError(
                f"Some of the parameters are not mapped. Unmapped parameters: {parameters}"
            )


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class BasicConfigSpace(ConfigSpace, ABC):
    @abc.abstractmethod
    def __post_init__(self) -> None:
        pass

    def _parse_fields(self, keys: Iterable[str]) -> Dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in keys}


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class AdvancedConfigSpace(ConfigSpace, ABC):
    @abc.abstractmethod
    def __post_init__(self) -> None:
        pass

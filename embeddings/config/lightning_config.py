from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Dict, List, Optional, Set, Union

from pytorch_lightning.accelerators import Accelerator

from embeddings.config.base_config import AdvancedConfig, BasicConfig
from embeddings.data.io import T_path
from embeddings.utils.loggers import get_logger
from embeddings.utils.utils import read_yaml

T_kwarg = MappingProxyType[str, Any]

_logger = get_logger(__name__)


@dataclass
class LightningConfigKeys:
    PIPELINE_KEYS: ClassVar[Set[str]] = {
        "batch_size",
        "finetune_last_n_layers",
        "embedding_name_or_path",
    }
    DATAMODULE_KEYS: ClassVar[Set[str]] = {"max_seq_length"}
    TASK_MODEL_KEYS: ClassVar[Set[str]] = {
        "learning_rate",
        "optimizer",
        "use_scheduler",
        "warmup_steps",
        "adam_epsilon",
        "weight_decay",
    }
    TASK_TRAIN_KEYS: ClassVar[Set[str]] = {"max_epochs", "devices", "accelerator"}
    MODEL_CONFIG_KEYS: ClassVar[Set[str]] = {"classifier_dropout"}
    EARLY_STOPPING_KEYS: ClassVar[Set[str]] = {"monitor", "mode", "patience"}


@dataclass
class LightningConfigDefaultKwargs:
    DEFAULT_DATAMODULE_KWARGS: ClassVar[T_kwarg] = MappingProxyType({"max_seq_length": None})
    DEFAULT_TASK_MODEL_KWARGS: ClassVar[T_kwarg] = MappingProxyType({"use_scheduler": True})
    DEFAULT_TASK_TRAIN_KWARGS: ClassVar[T_kwarg] = MappingProxyType(
        {
            "devices": "auto",
            "accelerator": "auto",
        }
    )
    DEFAULT_MODEL_CONFIG_KWARGS: ClassVar[T_kwarg] = MappingProxyType({"classifier_dropout": None})
    DEFAULT_EARLY_STOPPING_KWARGS: ClassVar[T_kwarg] = MappingProxyType(
        {
            "monitor": "val/Loss",
            "mode": "min",
            "patience": 3,
        }
    )


@dataclass
class LightningBasicConfig(BasicConfig, LightningConfigKeys):
    use_scheduler: bool = True
    optimizer: str = "Adam"
    warmup_steps: int = 100
    learning_rate: float = 1e-4
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    finetune_last_n_layers: int = -1
    classifier_dropout: Optional[float] = None
    max_seq_length: Optional[int] = None
    mini_batch_size: int = 32
    devices: Optional[Union[List[int], str, int]] = "auto"
    max_epochs: Optional[int] = None
    accelerator: Optional[Union[str, Accelerator]] = "auto"
    monitor: str = "val/Loss"
    mode: str = "min"
    patience: int = 3

    train_batch_size: int = field(init=False)
    eval_batch_size: int = field(init=False)
    datamodule_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    task_model_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    task_train_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    model_config_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    early_stopping_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    batch_encoding_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.train_batch_size = self.eval_batch_size = self.mini_batch_size
        self.datamodule_kwargs = self._parse_fields(self.DATAMODULE_KEYS)
        self.task_model_kwargs = self._parse_fields(self.TASK_MODEL_KEYS)
        self.task_train_kwargs = self._parse_fields(self.TASK_TRAIN_KEYS)
        self.model_config_kwargs = self._parse_fields(self.MODEL_CONFIG_KEYS)
        self.early_stopping_kwargs = self._parse_fields(self.EARLY_STOPPING_KEYS)
        self.task_model_kwargs.update(
            {"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}
        )

    @classmethod
    def from_yaml(cls, path: T_path) -> "LightningBasicConfig":
        config = read_yaml(path)
        return cls(**config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LightningBasicConfig":
        return cls(**config)


@dataclass
class LightningAdvancedConfig(AdvancedConfig, LightningConfigDefaultKwargs):
    train_batch_size: int = 32
    eval_batch_size: int = 32
    finetune_last_n_layers: int = -1
    datamodule_kwargs: Dict[str, Any] = field(default_factory=dict)
    task_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    task_train_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_config_kwargs: Dict[str, Any] = field(default_factory=dict)
    early_stopping_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    batch_encoding_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.datamodule_kwargs = {**self.DEFAULT_DATAMODULE_KWARGS, **self.datamodule_kwargs}
        self.task_model_kwargs = {**self.DEFAULT_TASK_MODEL_KWARGS, **self.task_model_kwargs}
        self.task_train_kwargs = {**self.DEFAULT_TASK_TRAIN_KWARGS, **self.task_train_kwargs}
        self.model_config_kwargs = {**self.DEFAULT_MODEL_CONFIG_KWARGS, **self.model_config_kwargs}
        self.early_stopping_kwargs = {
            **self.DEFAULT_EARLY_STOPPING_KWARGS,
            **self.early_stopping_kwargs,
        }
        self.task_model_kwargs.update(
            {"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}
        )

    @classmethod
    def from_yaml(cls, path: T_path) -> "LightningAdvancedConfig":
        config = read_yaml(path)
        return cls(**config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LightningAdvancedConfig":
        return cls(**config)


LightningConfig = Union[LightningBasicConfig, LightningAdvancedConfig]

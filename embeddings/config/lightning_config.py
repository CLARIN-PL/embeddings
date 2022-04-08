from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Set, Tuple, Union

from embeddings.config.base_config import AdvancedConfig, BasicConfig


@dataclass
class LightningConfigKeys:
    PIPELINE_KEYS: ClassVar[Set[str]] = {
        "finetune_last_n_layers",
        "embedding_name_or_path",
        "devices",
        "accelerator",
    }
    DATAMODULE_KEYS: ClassVar[Set[str]] = {"max_seq_length"}
    TASK_MODEL_KEYS: ClassVar[Set[str]] = {
        "learning_rate",
        "optimizer",
        "use_scheduler",
        "warmup_steps",
        "adam_epsilon",
        "weight_decay",
        "batch_size",
    }
    TASK_TRAIN_KEYS: ClassVar[Set[str]] = {"max_epochs"}
    MODEL_CONFIG_KEYS: ClassVar[Set[str]] = {"classifier_dropout"}
    EARLY_STOPPING_KEYS: ClassVar[Set[Tuple[str, str]]] = {
        ("early_stopping_monitor", "monitor"),
        ("early_stopping_mode", "mode"),
        ("early_stopping_patience", "patience"),
    }


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
    batch_size: int = 32
    max_epochs: Optional[int] = None
    early_stopping_monitor: str = "val/Loss"
    early_stopping_mode: str = "min"
    early_stopping_patience: int = 3

    tokenizer_kwargs: Dict[str, Any] = field(init=False, default_factory=dict)
    batch_encoding_kwargs: Dict[str, Any] = field(init=False, default_factory=dict)
    dataloader_kwargs: Dict[str, Any] = field(init=False, default_factory=dict)

    @property
    def train_batch_size(self) -> int:
        return self.batch_size

    @property
    def eval_batch_size(self) -> int:
        return self.batch_size

    @property
    def datamodule_kwargs(self) -> Dict[str, Any]:
        return self._parse_fields(self.DATAMODULE_KEYS)

    @property
    def task_model_kwargs(self) -> Dict[str, Any]:
        task_model_kwargs = self._parse_fields(self.TASK_MODEL_KEYS)
        task_model_kwargs.update(
            {"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}
        )
        return task_model_kwargs

    @property
    def task_train_kwargs(self) -> Dict[str, Any]:
        return self._parse_fields(self.TASK_TRAIN_KEYS)

    @property
    def model_config_kwargs(self) -> Dict[str, Any]:
        return self._parse_fields(self.MODEL_CONFIG_KEYS)

    @property
    def early_stopping_kwargs(self) -> Dict[str, Any]:
        return self._map_parse_fields(self.EARLY_STOPPING_KEYS)


@dataclass
class LightningAdvancedConfig(AdvancedConfig):
    finetune_last_n_layers: int
    task_model_kwargs: Dict[str, Any]
    datamodule_kwargs: Dict[str, Any]
    task_train_kwargs: Dict[str, Any]
    model_config_kwargs: Dict[str, Any]
    early_stopping_kwargs: Dict[str, Any]
    tokenizer_kwargs: Dict[str, Any]
    batch_encoding_kwargs: Dict[str, Any]
    dataloader_kwargs: Dict[str, Any]

    @staticmethod
    def from_basic() -> "LightningAdvancedConfig":
        config = LightningBasicConfig()
        return LightningAdvancedConfig(
            finetune_last_n_layers=config.finetune_last_n_layers,
            datamodule_kwargs=config.datamodule_kwargs,
            task_model_kwargs=config.task_model_kwargs,
            task_train_kwargs=config.task_train_kwargs,
            model_config_kwargs=config.model_config_kwargs,
            early_stopping_kwargs=config.early_stopping_kwargs,
            batch_encoding_kwargs=config.batch_encoding_kwargs,
            dataloader_kwargs=config.dataloader_kwargs,
            tokenizer_kwargs=config.tokenizer_kwargs,
        )


LightningConfig = Union[LightningBasicConfig, LightningAdvancedConfig]

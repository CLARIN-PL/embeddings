from abc import ABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Set, Tuple, Union

from embeddings.config.base_config import AdvancedConfig, BasicConfig
from embeddings.data.io import T_path
from embeddings.utils.utils import read_yaml


@dataclass
class FlairSequenceLabelingConfigKeys:
    TASK_MODEL_KEYS: ClassVar[Set[str]] = {
        "hidden_size",
        "use_crf",
        "use_rnn",
        "rnn_type",
        "rnn_layers",
        "dropout",
        "word_dropout",
        "locked_dropout",
        "reproject_embeddings",
    }


@dataclass
class FlairTextClassificationConfigMapping:
    TASK_MODEL_KEYS_MAPPING: ClassVar[MappingProxyType[str, Any]] = MappingProxyType(
        {
            "FlairDocumentCNNEmbeddings": {
                "cnn_pool_kernels",
                "dropout",
                "word_dropout",
                "reproject_words",
            },
            "FlairDocumentRNNEmbeddings": {
                "hidden_size",
                "rnn_type",
                "rnn_layers",
                "bidirectional",
                "dropout",
                "word_dropout",
                "reproject_words",
            },
            "FlairTransformerDocumentEmbedding": {"dynamic_pooling", "dynamic_fine_tune"},
            "FlairDocumentPoolEmbedding": {"static_pooling", "static_fine_tune_mode"},
        }
    )


@dataclass
class FlairBasicConfig(BasicConfig):
    TASK_TRAIN_KEYS: ClassVar[Set[str]] = {
        "learning_rate",
        "mini_batch_size",
        "max_epochs",
    }
    document_embedding_cls: str = "FlairDocumentPoolEmbedding"
    learning_rate: float = 1e-3
    mini_batch_size: int = 32
    max_epochs: int = 20

    task_model_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    task_train_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    load_dataset_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.task_train_kwargs = self._parse_fields(self.TASK_TRAIN_KEYS)

    @classmethod
    def from_yaml(cls, path: T_path) -> "FlairBasicConfig":
        config = read_yaml(path)
        return cls(**config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FlairBasicConfig":
        return cls(**config)


@dataclass
class FlairSequenceLabelingBasicConfig(FlairBasicConfig, FlairSequenceLabelingConfigKeys):
    hidden_size: int = 256
    use_crf: bool = True
    use_rnn: bool = True
    rnn_type: str = "LSTM"
    rnn_layers: int = 1
    dropout: float = 0.0
    word_dropout: float = 0.05
    locked_dropout: float = 0.5
    reproject_embeddings: bool = True

    def __post_init__(self) -> None:
        self.task_train_kwargs = self._parse_fields(self.TASK_TRAIN_KEYS)
        self.task_model_kwargs = self._parse_fields(self.TASK_MODEL_KEYS)

    @classmethod
    def from_yaml(cls, path: T_path) -> "FlairSequenceLabelingBasicConfig":
        pass

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlairSequenceLabelingBasicConfig":
        pass


@dataclass
class FlairTextClassificationBasicConfig(FlairBasicConfig, FlairTextClassificationConfigMapping):
    pooling: str = "mean"
    fine_tune_mode: str = "none"
    fine_tune: bool = False
    cnn_pool_kernels: Tuple[Tuple[int, int], ...] = ((100, 3), (100, 4), (100, 5))
    hidden_size: int = 256
    rnn_type: str = "LSTM"
    rnn_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.0
    word_dropout: float = 0.05
    reproject_words: bool = True

    load_model_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        task_model_keys = self.TASK_MODEL_KEYS_MAPPING[self.document_embedding_cls]
        self.task_train_kwargs = self._parse_fields(self.TASK_TRAIN_KEYS)
        self.task_model_kwargs = self._parse_fields(task_model_keys)

    @classmethod
    def from_yaml(cls, path: T_path) -> "FlairTextClassificationBasicConfig":
        pass

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlairTextClassificationBasicConfig":
        pass


@dataclass
class FlairAdvancedConfig(AdvancedConfig, ABC):
    DEFAULT_TASK_TRAIN_KWARGS: ClassVar[MappingProxyType[str, Any]] = MappingProxyType(
        {
            "learning_rate": 1e-3,
            "mini_batch_size": 32,
            "max_epochs": 20,
        }
    )
    document_embedding_cls: str = "FlairDocumentPoolEmbedding"

    task_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    task_train_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_model_kwargs = {**self.DEFAULT_TASK_TRAIN_KWARGS, **self.task_model_kwargs}

    @classmethod
    def from_yaml(cls, path: T_path) -> "FlairAdvancedConfig":
        config = read_yaml(path)
        return cls(**config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FlairAdvancedConfig":
        return cls(**config)


class FlairSequenceLabelingAdvancedConfig(FlairAdvancedConfig):
    hidden_size: int = 256


class FlairTextClassificationAdvancedConfig(FlairAdvancedConfig):
    load_model_kwargs: Dict[str, Any] = field(default_factory=dict)


FlairTextClassificationConfig = Union[
    FlairTextClassificationBasicConfig, FlairTextClassificationAdvancedConfig
]
FlairSequenceLabelingConfig = Union[
    FlairSequenceLabelingBasicConfig, FlairSequenceLabelingAdvancedConfig
]

from abc import ABC
from dataclasses import dataclass, field
from itertools import chain
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Set, Tuple, Union

from embeddings.config.base_config import AdvancedConfig, BasicConfig
from embeddings.data.io import T_path
from embeddings.utils.utils import read_yaml


@dataclass
class FlairConfigKeys:
    TASK_TRAIN_KEYS: ClassVar[Set[str]] = {
        "learning_rate",
        "mini_batch_size",
        "max_epochs",
    }


@dataclass
class FlairSequenceLabelingConfigKeys(FlairConfigKeys):
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
class FlairTextClassificationConfigKeys(FlairConfigKeys):
    CNN_EMBEDDING_KEYS: ClassVar[Set[str]] = {
        "cnn_pool_kernels",
        "dropout",
        "word_dropout",
        "reproject_words",
    }
    RNN_EMBEDDING_KEYS: ClassVar[Set[str]] = {
        "hidden_size",
        "rnn_type",
        "rnn_layers",
        "bidirectional",
        "dropout",
        "word_dropout",
        "reproject_words",
    }
    LOAD_MODEL_CONFIG_KEYS_MAPPING: ClassVar[MappingProxyType[str, Any]] = MappingProxyType(
        {
            "FlairDocumentCNNEmbeddings": RNN_EMBEDDING_KEYS,
            "FlairDocumentRNNEmbeddings": CNN_EMBEDDING_KEYS,
            "FlairTransformerDocumentEmbedding": {"pooling", "fine_tune"},
            "FlairDocumentPoolEmbedding": {"pooling", "fine_tune_mode"},
        }
    )
    LOAD_MODEL_CONFIG_SPACE_KEYS_MAPPING: ClassVar[MappingProxyType[str, Any]] = MappingProxyType(
        {
            "FlairDocumentCNNEmbeddings": RNN_EMBEDDING_KEYS,
            "FlairDocumentRNNEmbeddings": CNN_EMBEDDING_KEYS,
            "FlairTransformerDocumentEmbedding": {"dynamic_pooling", "dynamic_fine_tune"},
            "FlairDocumentPoolEmbedding": {"static_pooling", "static_fine_tune_mode"},
        }
    )
    LOAD_MODEL_CFG_KEYS: ClassVar[Set[str]] = set(chain(*LOAD_MODEL_CONFIG_KEYS_MAPPING.values()))


@dataclass
class FlairBasicConfig(BasicConfig, FlairConfigKeys):
    learning_rate: float = 1e-3
    mini_batch_size: int = 32
    max_epochs: int = 20

    task_model_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    task_train_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)

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


@dataclass
class FlairTextClassificationBasicConfig(FlairBasicConfig, FlairTextClassificationConfigKeys):
    document_embedding_cls: str = "FlairDocumentPoolEmbedding"
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

    load_model_kwargs: Dict[str, Any] = field(init=True, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        load_model_keys = self.LOAD_MODEL_CONFIG_KEYS_MAPPING[self.document_embedding_cls]
        self.task_train_kwargs = self._parse_fields(self.TASK_TRAIN_KEYS)
        self.load_model_kwargs = self._parse_fields(load_model_keys)


@dataclass
class FlairAdvancedConfig(AdvancedConfig, FlairConfigKeys, ABC):
    task_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    task_train_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    @classmethod
    def from_yaml(cls, path: T_path) -> "FlairAdvancedConfig":
        config = read_yaml(path)
        return cls(**config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FlairAdvancedConfig":
        return cls(**config)


@dataclass
class FlairSequenceLabelingAdvancedConfig(FlairAdvancedConfig):
    hidden_size: int = 256


@dataclass
class FlairTextClassificationAdvancedConfig(FlairAdvancedConfig):
    document_embedding_cls: str = "FlairDocumentPoolEmbedding"
    load_model_kwargs: Dict[str, Any] = field(default_factory=dict)


FlairTextClassificationConfig = Union[
    FlairTextClassificationBasicConfig, FlairTextClassificationAdvancedConfig
]
FlairSequenceLabelingConfig = Union[
    FlairSequenceLabelingBasicConfig, FlairSequenceLabelingAdvancedConfig
]

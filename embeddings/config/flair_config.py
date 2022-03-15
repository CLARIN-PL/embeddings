from abc import ABC
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Mapping, Set, Tuple, Union

from embeddings.config.base_config import AdvancedConfig, BasicConfig
from embeddings.data.io import T_path
from embeddings.utils.utils import read_yaml


@dataclass
class FlairTextClassificationConfigMapping:
    LOAD_MODEL_KEYS_MAPPING: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {
            "FlairDocumentCNNEmbeddings": {
                "hidden_size",
                "rnn_type",
                "rnn_layers",
                "bidirectional",
                "dropout",
                "word_dropout",
                "reproject_words",
            },
            "FlairDocumentRNNEmbeddings": {
                "cnn_pool_kernels",
                "dropout",
                "word_dropout",
                "reproject_words",
            },
            "FlairTransformerDocumentEmbedding": {"pooling", "fine_tune"},
            "FlairDocumentPoolEmbedding": {"pooling", "fine_tune_mode"},
        }
    )


@dataclass
class FlairBasicConfig(BasicConfig):
    learning_rate: float = 1e-3
    mini_batch_size: int = 32
    max_epochs: int = 20

    task_model_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    task_train_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.task_train_kwargs = self._parse_fields(self.get_config_keys())

    @classmethod
    def from_yaml(cls, path: T_path) -> "FlairBasicConfig":
        config = read_yaml(path)
        return cls(**config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FlairBasicConfig":
        return cls(**config)


@dataclass
class FlairSequenceLabelingBasicConfig(FlairBasicConfig):
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
        self.task_model_kwargs = self._parse_fields(self.get_config_keys())
        super().__post_init__()


@dataclass
class FlairTextClassificationBasicConfig(FlairBasicConfig, FlairTextClassificationConfigMapping):
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
        self.load_model_kwargs = self._parse_fields(self.get_config_keys())
        super().__post_init__()

    def get_config_keys(self) -> Set[str]:
        return self.LOAD_MODEL_KEYS_MAPPING[self.document_embedding_cls]


@dataclass
class FlairAdvancedConfig(AdvancedConfig, ABC):
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

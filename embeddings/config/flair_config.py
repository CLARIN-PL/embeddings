from abc import ABC
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Mapping, Set, Tuple, Union

from embeddings.config.base_config import AdvancedConfig, BasicConfig


@dataclass
class FlairTextClassificationConfigMapping:
    LOAD_MODEL_KEYS_MAPPING: ClassVar[Mapping[str, Set[str]]] = MappingProxyType(
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

    @classmethod
    def map_load_model_keys(cls, document_embedding_cls: str) -> Set[str]:
        return cls.LOAD_MODEL_KEYS_MAPPING[document_embedding_cls]

    @classmethod
    def get_load_model_keys(cls) -> Set[str]:
        return set(chain(*cls.LOAD_MODEL_KEYS_MAPPING.values()))


@dataclass
class FlairBasicConfig(BasicConfig, ABC):
    learning_rate: float = 1e-3
    mini_batch_size: int = 32
    max_epochs: int = 20

    @property
    def task_train_kwargs(self) -> Dict[str, Any]:
        return self._parse_fields(self.get_task_train_keys())

    @staticmethod
    def get_task_train_keys() -> Set[str]:
        task_train_keys = {
            key for key in FlairBasicConfig.__annotations__.keys() if not key.endswith("_kwargs")
        }
        return task_train_keys


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

    @property
    def task_model_kwargs(self) -> Dict[str, Any]:
        return self._parse_fields(self.get_task_model_keys())

    @property
    def load_model_kwargs(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_task_model_keys(cls) -> Set[str]:
        task_model_keys = set(cls.__annotations__.keys())
        task_model_keys.remove("hidden_size")
        return task_model_keys


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

    @property
    def load_model_kwargs(self) -> Dict[str, Any]:
        load_model_keys = self.get_map_load_model_keys(self.document_embedding_cls)
        return self._parse_fields(load_model_keys)

    @property
    def task_model_kwargs(self) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_map_load_model_keys(cls, document_embedding_cls: str) -> Set[str]:
        return cls.map_load_model_keys(document_embedding_cls)


@dataclass
class FlairAdvancedConfig(AdvancedConfig, ABC):
    task_model_kwargs: Dict[str, Any]
    task_train_kwargs: Dict[str, Any]
    load_model_kwargs: Dict[str, Any]


@dataclass
class FlairSequenceLabelingAdvancedConfig(FlairAdvancedConfig):
    hidden_size: int

    @classmethod
    def from_basic(cls) -> "FlairSequenceLabelingAdvancedConfig":
        config = FlairSequenceLabelingBasicConfig()
        return cls(
            hidden_size=config.hidden_size,
            task_model_kwargs=config.task_model_kwargs,
            task_train_kwargs=config.task_train_kwargs,
            load_model_kwargs=config.load_model_kwargs,
        )


@dataclass
class FlairTextClassificationAdvancedConfig(FlairAdvancedConfig):
    document_embedding_cls: str

    @classmethod
    def from_basic(cls) -> "FlairTextClassificationAdvancedConfig":
        config = FlairTextClassificationBasicConfig()
        return cls(
            document_embedding_cls=config.document_embedding_cls,
            task_model_kwargs=config.task_model_kwargs,
            task_train_kwargs=config.task_train_kwargs,
            load_model_kwargs=config.load_model_kwargs,
        )


FlairTextClassificationConfig = Union[
    FlairTextClassificationBasicConfig, FlairTextClassificationAdvancedConfig
]
FlairSequenceLabelingConfig = Union[
    FlairSequenceLabelingBasicConfig, FlairSequenceLabelingAdvancedConfig
]

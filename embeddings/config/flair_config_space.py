from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Set, Tuple

from embeddings.config.config_space import BasicConfigSpace
from embeddings.data.io import T_path


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
class FlairBasicConfigSpace(BasicConfigSpace):
    TASK_TRAIN_KEYS: ClassVar[Set[str]] = {
        "learning_rate",
        "mini_batch_size",
        "max_epochs",
    }

    learning_rate: float = 1e-3
    mini_batch_size: int = 32
    max_epochs: int = 20

    task_model_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    task_train_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)
    load_dataset_kwargs: Dict[str, Any] = field(init=False, compare=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.task_train_kwargs = self._parse_fields(self.TASK_TRAIN_KEYS)

    @classmethod
    def from_yaml(cls, path: T_path) -> "FlairBasicConfigSpace":
        pass

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlairBasicConfigSpace":
        pass


@dataclass
class FlairAdvancedConfigSpace(BasicConfigSpace):
    DEFAULT_TASK_TRAIN_KWARGS: ClassVar[MappingProxyType[str, Any]] = MappingProxyType(
        {
            "learning_rate": 1e-3,
            "mini_batch_size": 32,
            "max_epochs": 20,
        }
    )

    task_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    task_train_kwargs: Dict[str, Any] = field(default_factory=dict)
    load_dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    load_model_kwargs: Dict[str, Any] = field(default_factory=dict)  # classification task
    hidden_size: int = field(init=False)  # sequence labeling task

    def __post_init__(self) -> None:
        self.task_model_kwargs = {**self.DEFAULT_TASK_TRAIN_KWARGS, **self.task_model_kwargs}
        if "hidden_size" in self.task_model_kwargs:
            self.hidden_size = self.task_model_kwargs["hidden_size"]

    @classmethod
    def from_yaml(cls, path: T_path) -> "FlairAdvancedConfigSpace":
        pass

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlairAdvancedConfigSpace":
        pass


@dataclass
class FlairSequenceLabelingBasicConfigSpace(FlairBasicConfigSpace, FlairSequenceLabelingConfigKeys):
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
    def from_yaml(cls, path: T_path) -> "FlairSequenceLabelingBasicConfigSpace":
        pass

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlairSequenceLabelingBasicConfigSpace":
        pass


@dataclass
class FlairTextClassificationBasicConfigSpace(
    FlairBasicConfigSpace, FlairTextClassificationConfigMapping
):
    document_embedding: str = "FlairDocumentPoolEmbedding"
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

    def __post_init__(self) -> None:
        task_model_keys = self.TASK_MODEL_KEYS_MAPPING[self.document_embedding]
        self.task_train_kwargs = self._parse_fields(self.TASK_TRAIN_KEYS)
        self.task_model_kwargs = self._parse_fields(task_model_keys)

    @classmethod
    def from_yaml(cls, path: T_path) -> "FlairTextClassificationBasicConfigSpace":
        pass

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlairTextClassificationBasicConfigSpace":
        pass

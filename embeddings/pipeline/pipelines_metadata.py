from typing import Any, Dict, Optional, Tuple, TypeVar

from typing_extensions import Literal, TypedDict

from embeddings.data.io import T_path


class PathMetadata(TypedDict, total=False):
    output_path: T_path


class EmbeddingPipelineMetadata(PathMetadata):
    embedding_name: str
    dataset_name: str
    load_dataset_kwargs: Optional[Dict[str, Any]]
    task_model_kwargs: Optional[Dict[str, Any]]
    task_train_kwargs: Optional[Dict[str, Any]]


class HuggingFaceClassificationPipelineMetadata(EmbeddingPipelineMetadata):
    input_column_name: str
    target_column_name: str
    document_embedding_cls: str
    load_model_kwargs: Optional[Dict[str, Any]]


class HuggingFacePairClassificationPipelineMetadata(EmbeddingPipelineMetadata):
    input_columns_names_pair: Tuple[str, str]
    target_column_name: str
    document_embedding_cls: str
    load_model_kwargs: Optional[Dict[str, Any]]


class HuggingFaceSequenceLabelingPipelineMetadata(EmbeddingPipelineMetadata):
    input_column_name: str
    target_column_name: str
    hidden_size: int
    evaluation_mode: str
    tagging_scheme: Optional[str]


class EvaluationPipelineMetadata(TypedDict):
    dataset_path: str
    embedding_name: str
    persist_path: Optional[str]
    predict_subset: Literal["dev", "test"]
    task_model_kwargs: Optional[Dict[str, Any]]
    task_train_kwargs: Optional[Dict[str, Any]]
    output_path: str


class FlairSequenceLabelingEvaluationPipelineMetadata(EvaluationPipelineMetadata):
    hidden_size: int
    evaluation_mode: str
    tagging_scheme: Optional[str]


class FlairClassificationEvaluationPipelineMetadata(EvaluationPipelineMetadata):
    document_embedding_cls: str
    load_model_kwargs: Optional[Dict[str, Any]]


Metadata = TypeVar("Metadata", bound=EmbeddingPipelineMetadata)
EvaluationMetadata = TypeVar("EvaluationMetadata", bound=EvaluationPipelineMetadata)

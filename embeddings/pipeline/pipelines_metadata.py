from typing import Any, Dict, Optional, Tuple, TypedDict, TypeVar

from embeddings.data.io import T_path

Metadata = TypeVar("Metadata")


class PathMetadata(TypedDict, total=False):
    output_path: T_path


class EmbeddingPipelineMetadata(PathMetadata):
    embedding_name: str
    dataset_name: str
    task_model_kwargs: Optional[Dict[str, Any]]
    task_train_kwargs: Optional[Dict[str, Any]]


class HuggingFaceClassificationPipelineMetadata(EmbeddingPipelineMetadata):
    input_column_name: str
    target_column_name: str


class HuggingFacePairClassificationPipelineMetadata(EmbeddingPipelineMetadata):
    input_columns_names_pair: Tuple[str, str]
    target_column_name: str


class HuggingFaceSequenceLabelingPipelineMetadata(EmbeddingPipelineMetadata):
    input_column_name: str
    target_column_name: str
    hidden_size: int
    evaluation_mode: str
    tagging_scheme: Optional[str]

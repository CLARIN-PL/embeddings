from typing import Any, Dict, Optional, Tuple, TypeVar

from typing_extensions import Literal, TypedDict

from embeddings.data.io import T_path


class PathMetadata(TypedDict, total=False):
    output_path: T_path


class EmbeddingPipelineBaseMetadata(PathMetadata):
    embedding_name: str
    task_model_kwargs: Optional[Dict[str, Any]]
    task_train_kwargs: Optional[Dict[str, Any]]


class FlairEmbeddingPipelineMetadata(EmbeddingPipelineBaseMetadata):
    dataset_name: str
    load_dataset_kwargs: Optional[Dict[str, Any]]


class FlairClassificationPipelineMetadata(FlairEmbeddingPipelineMetadata):
    input_column_name: str
    target_column_name: str
    document_embedding_cls: str
    load_model_kwargs: Optional[Dict[str, Any]]


class FlairPairClassificationPipelineMetadata(FlairEmbeddingPipelineMetadata):
    input_columns_names_pair: Tuple[str, str]
    target_column_name: str
    document_embedding_cls: str
    load_model_kwargs: Optional[Dict[str, Any]]


class FlairSequenceLabelingPipelineMetadata(FlairEmbeddingPipelineMetadata):
    input_column_name: str
    target_column_name: str
    hidden_size: int
    evaluation_mode: str
    tagging_scheme: Optional[str]


class FlairEvaluationPipelineMetadata(EmbeddingPipelineBaseMetadata):
    dataset_path: str
    persist_path: Optional[str]
    predict_subset: Literal["dev", "test"]


class FlairSequenceLabelingEvaluationPipelineMetadata(FlairEvaluationPipelineMetadata):
    hidden_size: int
    evaluation_mode: str
    tagging_scheme: Optional[str]


class FlairClassificationEvaluationPipelineMetadata(FlairEvaluationPipelineMetadata):
    document_embedding_cls: str
    load_model_kwargs: Optional[Dict[str, Any]]


class LightningClassificationPipelineMetadata(EmbeddingPipelineBaseMetadata):
    dataset_name_or_path: str
    input_column_name: str
    target_column_name: str
    train_batch_size: int
    eval_batch_size: int
    finetune_last_n_layers: int
    tokenizer_name: Optional[str]
    load_dataset_kwargs: Optional[Dict[str, Any]]
    datamodule_kwargs: Optional[Dict[str, Any]]
    tokenizer_kwargs: Optional[Dict[str, Any]]
    batch_encoding_kwargs: Optional[Dict[str, Any]]
    predict_subset: Literal["dev", "test"]


Metadata = TypeVar("Metadata", bound=EmbeddingPipelineBaseMetadata)
EvaluationMetadata = TypeVar("EvaluationMetadata", bound=EmbeddingPipelineBaseMetadata)

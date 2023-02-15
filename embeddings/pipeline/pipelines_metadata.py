from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

from pytorch_lightning.accelerators import Accelerator
from typing_extensions import Literal, TypedDict

from embeddings.config.lightning_config import LightningConfig
from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.metric.sequence_labeling import EvaluationMode, TaggingScheme


class EmbeddingPipelineBaseMetadata(TypedDict, total=False):
    output_path: T_path


class LightningPipelineMetadata(EmbeddingPipelineBaseMetadata):
    embedding_name_or_path: T_path
    dataset_name_or_path: T_path
    target_column_name: str
    config: LightningConfig
    devices: Optional[Union[List[int], str, int]]
    accelerator: Optional[Union[str, Accelerator]]
    tokenizer_name_or_path: Optional[T_path]
    load_dataset_kwargs: Optional[Dict[str, Any]]
    predict_subset: Literal[LightingDataModuleSubset.VALIDATION, LightingDataModuleSubset.TEST]


class LightningClassificationPipelineMetadata(LightningPipelineMetadata):
    input_column_name: Union[str, Sequence[str]]


class LightningSequenceLabelingPipelineMetadata(LightningPipelineMetadata):
    input_column_name: str
    evaluation_mode: EvaluationMode
    tagging_scheme: Optional[TaggingScheme]


Metadata = TypeVar("Metadata", bound=EmbeddingPipelineBaseMetadata)
LightningMetadata = TypeVar("LightningMetadata", bound=LightningPipelineMetadata)
EvaluationMetadata = TypeVar("EvaluationMetadata", bound=EmbeddingPipelineBaseMetadata)

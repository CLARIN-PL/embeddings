from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Generic, Dict, Any, Tuple

from embeddings.hyperparameter_search.configspace import SampledParameters, \
    ConfigSpace
from embeddings.hyperparameter_search.lighting_configspace import (
    LightingTextClassificationConfigSpace,
)
from embeddings.pipeline.hf_preprocessing_pipeline import \
    HuggingFaceTextClassificationPreprocessingPipeline
from embeddings.pipeline.hps_pipeline import OptunaPipeline, \
    AbstractHuggingFaceOptimizedPipeline, _HuggingFaceOptimizedPipelineBase
from embeddings.pipeline.lightning_classification import \
    LightningClassificationPipeline
from embeddings.pipeline.pipelines_metadata import LightningClassificationPipelineMetadata
from embeddings.utils.utils import PrimitiveTypes


@dataclass
class _OptimizedLightingPipelineBase(_HuggingFaceOptimizedPipelineBase[ConfigSpace], ABC, Generic[ConfigSpace]):
    input_column_name: str
    target_column_name: str

    tokenizer_name: Optional[str] = field(default=None)
    dataset_dir: TemporaryDirectory[str] = field(init=False, default_factory=TemporaryDirectory)
    tmp_model_output_dir: TemporaryDirectory[str] = field(
        init=False, default_factory=TemporaryDirectory
    )
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    batch_encoding_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class OptimizedLightingClassificationPipeline(
    _OptimizedLightingPipelineBase,
    OptunaPipeline[
        LightingTextClassificationConfigSpace,
        LightningClassificationPipelineMetadata,
        LightningClassificationPipelineMetadata,
    ],
    AbstractHuggingFaceOptimizedPipeline[LightingTextClassificationConfigSpace],
):

    #
    # def __post_init__(self) -> None:
    #     super().__init__(
    #         preprocessing_pipeline=HuggingFaceTextClassificationPreprocessingPipeline(
    #             dataset_name=self.dataset_name,
    #             persist_path=self.dataset_dir.name,
    #             sample_missing_splits=(self.sample_dev_split_fraction, None),
    #             ignore_test_subset=True,
    #             load_dataset_kwargs=self.load_dataset_kwargs,
    #         ),
    #         evaluation_pipeline=LightningClassificationPipeline,
    #         pruner=self.pruner_cls(n_warmup_steps=self.n_warmup_steps),
    #         sampler=self.sampler_cls(seed=self.seed),
    #         n_trials=self.n_trials,
    #         dataset_path=self.dataset_path,
    #         metric_name="f1__average_macro",
    #         metric_key="f1",
    #         config_space=self.config_space,
    #     )
    #
    # def _get_metadata(self, parameters: SampledParameters) -> LightningClassificationPipelineMetadata:
    #     (
    #         embedding_name,
    #         document_embedding_cls,
    #         task_train_kwargs,
    #         load_model_kwargs,
    #     ) = self._pop_sampled_parameters(parameters=parameters)
    #     metadata: LightningClassificationPipelineMetadata = {
    #         "embedding_name": embedding_name,
    #         "document_embedding_cls": document_embedding_cls,
    #         "dataset_name": self.dataset_name,
    #         "load_dataset_kwargs": self.load_dataset_kwargs,
    #         "input_column_name": self.input_column_name,
    #         "target_column_name": self.target_column_name,
    #         "task_model_kwargs": None,
    #         "task_train_kwargs": task_train_kwargs,
    #         "load_model_kwargs": load_model_kwargs,
    #     }
    #     return metadata
    #
    # def _get_evaluation_metadata(
    #     self, parameters: SampledParameters
    # ) -> LightningClassificationPipelineMetadata:
    #     (
    #         embedding_name,
    #         train_batch_size,
    #         eval_batch_size,
    #         unfreeze_from,
    #         datamodule_kwargs,
    #         task_model_kwargs,
    #         task_trainer_kwargs
    #     ) = self._pop_sampled_parameters(parameters=parameters)
    #     metadata: LightningClassificationPipelineMetadata = {
    #         "embedding_name": self.
    #         "dataset_name_or_path": self.dataset_dir.name,
    #         "document_embedding_cls": document_embedding_cls,
    #         "dataset_path": str(self.dataset_path),
    #         "persist_path": None,
    #         "predict_subset": "dev",
    #         "output_path": self.tmp_model_output_dir.name,
    #         "task_model_kwargs": None,
    #         "task_train_kwargs": task_train_kwargs,
    #         "load_model_kwargs": load_model_kwargs,
    #     }
    #     return metadata

    def _post_run_hook(self) -> None:
        super()._post_run_hook()
        self.dataset_dir.cleanup()
        self.tmp_model_output_dir.cleanup()

    @staticmethod
    def _pop_sampled_parameters(
        parameters: SampledParameters,
    ) -> Tuple[str, int, int, Optional[int], Dict[str, PrimitiveTypes], Dict[str, PrimitiveTypes], Dict[str, PrimitiveTypes]]:
        embedding_name = parameters["embedding_name"]
        assert isinstance(embedding_name, str)
        train_batch_size = parameters["train_batch_size"]
        assert isinstance(train_batch_size, int)
        eval_batch_size = parameters["eval_batch_size"]
        assert isinstance(eval_batch_size, int)
        unfreeze_from = parameters["train_parameters"]
        assert isinstance(unfreeze_from, int) or unfreeze_from is None
        datamodule_kwargs = parameters["datamodule_kwargs"]
        assert isinstance(datamodule_kwargs, dict)
        task_model_kwargs = parameters["task_model_kwargs"]
        assert isinstance(task_model_kwargs, dict)
        task_trainer_kwargs = parameters["task_train_kwargs"]
        assert isinstance(task_trainer_kwargs, dict)

        return embedding_name, train_batch_size, eval_batch_size, unfreeze_from, datamodule_kwargs, task_model_kwargs, task_trainer_kwargs

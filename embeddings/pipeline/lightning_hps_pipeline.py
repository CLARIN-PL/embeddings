from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generic, Optional, Tuple

import datasets

from embeddings.data.dataset import LightingDataModuleSubset
from embeddings.data.io import T_path
from embeddings.evaluator.sequence_labeling_evaluator import (
    EvaluationMode,
    SequenceLabelingEvaluator,
    TaggingScheme,
)
from embeddings.hyperparameter_search.configspace import ConfigSpace, SampledParameters
from embeddings.hyperparameter_search.lighting_configspace import (
    LightingSequenceLabelingConfigSpace,
    LightingTextClassificationConfigSpace,
)
from embeddings.hyperparameter_search.parameters import ParameterValues
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.hps_pipeline import (
    AbstractHuggingFaceOptimizedPipeline,
    OptunaPipeline,
    _HuggingFaceOptimizedPipelineBase,
)
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline
from embeddings.pipeline.pipelines_metadata import (
    LightningClassificationPipelineMetadata,
    LightningMetadata,
    LightningSequenceLabelingPipelineMetadata,
)


@dataclass
class _OptimizedLightingPipelineBase(
    _HuggingFaceOptimizedPipelineBase[ConfigSpace], ABC, Generic[ConfigSpace]
):
    input_column_name: str
    target_column_name: str

    tmp_dataset_dir: TemporaryDirectory[str] = field(init=False, default_factory=TemporaryDirectory)
    tmp_model_output_dir: TemporaryDirectory[str] = field(
        init=False, default_factory=TemporaryDirectory
    )
    tokenizer_name_or_path: Optional[T_path] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    batch_encoding_kwargs: Optional[Dict[str, Any]] = None


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class OptimizedLightingPipeline(
    OptunaPipeline[
        ConfigSpace,
        LightningMetadata,
        LightningMetadata,
        str,
        datasets.DatasetDict,
        datasets.DatasetDict,
    ],
    AbstractHuggingFaceOptimizedPipeline[ConfigSpace],
    _OptimizedLightingPipelineBase[ConfigSpace],
    ABC,
    Generic[ConfigSpace, LightningMetadata],
):
    def _get_evaluation_metadata(self, parameters: SampledParameters) -> LightningMetadata:
        metadata = self._get_metadata(parameters)
        metadata["predict_subset"] = LightingDataModuleSubset.VALIDATION
        metadata["dataset_name_or_path"] = str(self.dataset_path)
        metadata["output_path"] = self.tmp_model_output_dir.name
        return metadata

    def _init_dataset_path(self) -> None:
        self.dataset_path: T_path
        if self.ignore_preprocessing_pipeline:
            self.dataset_path = Path(self.dataset_name_or_path)
            if not self.dataset_path.exists():
                raise FileNotFoundError("Dataset path not found")
        else:
            self.dataset_path = self.tmp_dataset_dir.name

    def _init_preprocessing_pipeline(self) -> None:
        self.preprocessing_pipeline: Optional[HuggingFacePreprocessingPipeline]
        if self.ignore_preprocessing_pipeline:
            self.preprocessing_pipeline = None
        else:
            self.preprocessing_pipeline = HuggingFacePreprocessingPipeline(
                dataset_name=str(self.dataset_name_or_path),
                persist_path=self.tmp_dataset_dir.name,
                sample_missing_splits=(self.sample_dev_split_fraction, None),
                ignore_test_subset=True,
                load_dataset_kwargs=self.load_dataset_kwargs,
            )

    @staticmethod
    def _pop_sampled_parameters(
        parameters: SampledParameters,
    ) -> Tuple[
        str,
        int,
        int,
        int,
        Dict[str, ParameterValues],
        Dict[str, ParameterValues],
        Dict[str, ParameterValues],
        Dict[str, ParameterValues],
    ]:
        embedding_name_or_path = parameters["embedding_name_or_path"]
        assert isinstance(embedding_name_or_path, str)
        train_batch_size = parameters["train_batch_size"]
        assert isinstance(train_batch_size, int)
        eval_batch_size = parameters["eval_batch_size"]
        assert isinstance(eval_batch_size, int)
        finetune_last_n_layers = parameters["finetune_last_n_layers"]
        assert isinstance(finetune_last_n_layers, int)
        datamodule_kwargs = parameters["datamodule_kwargs"]
        assert isinstance(datamodule_kwargs, dict)
        task_model_kwargs = parameters["task_model_kwargs"]
        assert isinstance(task_model_kwargs, dict)
        task_train_kwargs = parameters["task_train_kwargs"]
        assert isinstance(task_train_kwargs, dict)
        model_config_kwargs = parameters["task_train_kwargs"]
        assert isinstance(model_config_kwargs, dict)

        return (
            embedding_name_or_path,
            train_batch_size,
            eval_batch_size,
            finetune_last_n_layers,
            datamodule_kwargs,
            task_model_kwargs,
            task_train_kwargs,
            model_config_kwargs,
        )

    def _post_run_hook(self) -> None:
        super()._post_run_hook()
        self.tmp_dataset_dir.cleanup()
        self.tmp_model_output_dir.cleanup()


@dataclass
class OptimizedLightingClassificationPipeline(
    OptimizedLightingPipeline[
        LightingTextClassificationConfigSpace, LightningClassificationPipelineMetadata
    ]
):
    def __post_init__(self) -> None:
        # Type: ignore is temporal solution due to issue #152 https://github.com/CLARIN-PL/embeddings/issues/152
        self._init_dataset_path()
        self._init_preprocessing_pipeline()
        super(OptimizedLightingPipeline, self).__init__(
            preprocessing_pipeline=self.preprocessing_pipeline,
            evaluation_pipeline=LightningClassificationPipeline,  # type: ignore
            pruner=self.pruner_cls(n_warmup_steps=self.n_warmup_steps),
            sampler=self.sampler_cls(seed=self.seed),
            n_trials=self.n_trials,
            dataset_path=self.dataset_path,
            metric_name="f1__average_macro",
            metric_key="f1",
            config_space=self.config_space,
        )

    def _get_metadata(
        self, parameters: SampledParameters
    ) -> LightningClassificationPipelineMetadata:
        (
            embedding_name_or_path,
            train_batch_size,
            eval_batch_size,
            finetune_last_n_layers,
            datamodule_kwargs,
            task_model_kwargs,
            task_train_kwargs,
            model_config_kwargs,
        ) = self._pop_sampled_parameters(parameters=parameters)
        metadata: LightningClassificationPipelineMetadata = {
            "embedding_name_or_path": embedding_name_or_path,
            "dataset_name_or_path": self.dataset_name_or_path,
            "input_column_name": self.input_column_name,
            "target_column_name": self.target_column_name,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "finetune_last_n_layers": finetune_last_n_layers,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "datamodule_kwargs": datamodule_kwargs,
            "tokenizer_kwargs": self.tokenizer_kwargs,
            "batch_encoding_kwargs": self.batch_encoding_kwargs,
            "load_dataset_kwargs": self.load_dataset_kwargs,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
            "model_config_kwargs": model_config_kwargs,
            "predict_subset": LightingDataModuleSubset.TEST,
        }
        return metadata


@dataclass
class OptimizedLightingSequenceLabelingPipeline(
    OptimizedLightingPipeline[
        LightingSequenceLabelingConfigSpace,
        LightningSequenceLabelingPipelineMetadata,
    ]
):
    evaluation_mode: EvaluationMode = EvaluationMode.CONLL
    tagging_scheme: Optional[TaggingScheme] = None

    def __post_init__(self) -> None:
        self._init_dataset_path()
        self._init_preprocessing_pipeline()
        self.metric_name = SequenceLabelingEvaluator.get_metric_name(
            evaluation_mode=self.evaluation_mode, tagging_scheme=self.tagging_scheme
        )
        # Type: ignore is temporal solution due to issue #152 https://github.com/CLARIN-PL/embeddings/issues/152
        super(OptimizedLightingPipeline, self).__init__(
            preprocessing_pipeline=self.preprocessing_pipeline,
            evaluation_pipeline=LightningSequenceLabelingPipeline,  # type: ignore
            pruner=self.pruner_cls(n_warmup_steps=self.n_warmup_steps),
            sampler=self.sampler_cls(seed=self.seed),
            n_trials=self.n_trials,
            dataset_path=self.dataset_path,
            metric_name=self.metric_name,
            metric_key="overall_f1",
            config_space=self.config_space,
        )

    def _get_metadata(
        self, parameters: SampledParameters
    ) -> LightningSequenceLabelingPipelineMetadata:
        (
            embedding_name_or_path,
            train_batch_size,
            eval_batch_size,
            finetune_last_n_layers,
            datamodule_kwargs,
            task_model_kwargs,
            task_train_kwargs,
            model_config_kwargs,
        ) = self._pop_sampled_parameters(parameters=parameters)
        metadata: LightningSequenceLabelingPipelineMetadata = {
            "embedding_name_or_path": embedding_name_or_path,
            "dataset_name_or_path": self.dataset_name_or_path,
            "input_column_name": self.input_column_name,
            "target_column_name": self.target_column_name,
            "evaluation_mode": self.evaluation_mode,
            "tagging_scheme": self.tagging_scheme,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "finetune_last_n_layers": finetune_last_n_layers,
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "datamodule_kwargs": datamodule_kwargs,
            "tokenizer_kwargs": self.tokenizer_kwargs,
            "batch_encoding_kwargs": self.batch_encoding_kwargs,
            "load_dataset_kwargs": self.load_dataset_kwargs,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
            "model_config_kwargs": model_config_kwargs,
            "predict_subset": LightingDataModuleSubset.TEST,
        }
        return metadata

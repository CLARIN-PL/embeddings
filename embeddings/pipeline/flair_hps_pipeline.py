from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Generic, Optional, Tuple

import datasets
from flair.data import Corpus

from embeddings.data.io import T_path
from embeddings.evaluator.sequence_labeling_evaluator import (
    EvaluationMode,
    SequenceLabelingEvaluator,
    TaggingScheme,
)
from embeddings.hyperparameter_search.configspace import ConfigSpace, SampledParameters
from embeddings.hyperparameter_search.flair_configspace import (
    FlairSequenceLabelingConfigSpace,
    FlairTextClassificationConfigSpace,
)
from embeddings.hyperparameter_search.parameters import ParameterValues
from embeddings.pipeline.evaluation_pipeline import (
    FlairSequenceLabelingEvaluationPipeline,
    FlairTextClassificationEvaluationPipeline,
    FlairTextPairClassificationEvaluationPipeline,
)
from embeddings.pipeline.hps_pipeline import (
    AbstractHuggingFaceOptimizedPipeline,
    OptunaPipeline,
    _HuggingFaceOptimizedPipelineBase,
    _HuggingFaceOptimizedPipelineDefaultsBase,
)
from embeddings.pipeline.pipelines_metadata import (
    FlairClassificationEvaluationPipelineMetadata,
    FlairClassificationPipelineMetadata,
    FlairEvaluationPipelineMetadata,
    FlairPairClassificationPipelineMetadata,
    FlairSequenceLabelingEvaluationPipelineMetadata,
    FlairSequenceLabelingPipelineMetadata,
)
from embeddings.pipeline.preprocessing_pipeline import (
    FlairSequenceLabelingPreprocessingPipeline,
    FlairTextClassificationPreprocessingPipeline,
    FlairTextPairClassificationPreprocessingPipeline,
)


@dataclass
class _OptimizedFlairPipelineBase(
    _HuggingFaceOptimizedPipelineBase[ConfigSpace], ABC, Generic[ConfigSpace]
):
    input_column_name: str
    target_column_name: str


@dataclass
class _OptimizedFlairPairClassificationPipelineBase(
    _HuggingFaceOptimizedPipelineBase[ConfigSpace], ABC, Generic[ConfigSpace]
):
    input_columns_names_pair: Tuple[str, str]
    target_column_name: str


@dataclass
class _OptimizedFlairPipelineDefaultsBase(_HuggingFaceOptimizedPipelineDefaultsBase, ABC):
    tmp_dataset_dir: TemporaryDirectory[str] = field(init=False, default_factory=TemporaryDirectory)
    tmp_model_output_dir: TemporaryDirectory[str] = field(
        init=False, default_factory=TemporaryDirectory
    )


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class AbstractOptimizedFlairClassificationPipeline(
    AbstractHuggingFaceOptimizedPipeline[FlairTextClassificationConfigSpace],
    _OptimizedFlairPipelineDefaultsBase,
    ABC,
):
    def _init_dataset_path(self) -> None:
        self.dataset_path: T_path
        if self.ignore_preprocessing_pipeline:
            self.dataset_path = Path(self.dataset_name_or_path)
            if not self.dataset_path.exists():
                raise FileNotFoundError("Dataset path not found")
        else:
            self.dataset_path = Path(self.tmp_dataset_dir.name).joinpath("ds.pkl")

    @staticmethod
    def _pop_sampled_parameters(
        parameters: SampledParameters,
    ) -> Tuple[str, str, Dict[str, ParameterValues], Dict[str, ParameterValues]]:
        embedding_name = parameters["embedding_name"]
        assert isinstance(embedding_name, str)
        document_embedding = parameters["document_embedding"]
        assert isinstance(document_embedding, str)
        task_train_kwargs = parameters["task_train_kwargs"]
        assert isinstance(task_train_kwargs, dict)
        load_model_kwargs = parameters["load_model_kwargs"]
        assert isinstance(load_model_kwargs, dict)
        return embedding_name, document_embedding, task_train_kwargs, load_model_kwargs


@dataclass
class OptimizedFlairClassificationPipeline(
    OptunaPipeline[
        FlairTextClassificationConfigSpace,
        FlairClassificationPipelineMetadata,
        FlairEvaluationPipelineMetadata,
        str,
        datasets.DatasetDict,
        Corpus,
    ],
    AbstractOptimizedFlairClassificationPipeline,
    _OptimizedFlairPipelineBase[FlairTextClassificationConfigSpace],
):
    def _init_preprocessing_pipeline(self) -> None:
        self.preprocessing_pipeline: Optional[FlairTextClassificationPreprocessingPipeline]
        if self.ignore_preprocessing_pipeline:
            self.preprocessing_pipeline = None
        else:
            self.preprocessing_pipeline = FlairTextClassificationPreprocessingPipeline(
                dataset_name=str(self.dataset_name_or_path),
                input_column_name=self.input_column_name,
                target_column_name=self.target_column_name,
                persist_path=str(self.dataset_path),
                sample_missing_splits=(self.sample_dev_split_fraction, None),
                ignore_test_subset=True,
                load_dataset_kwargs=self.load_dataset_kwargs,
            )

    def __post_init__(self) -> None:
        self._init_dataset_path()
        self._init_preprocessing_pipeline()
        super().__init__(
            preprocessing_pipeline=self.preprocessing_pipeline,
            evaluation_pipeline=FlairTextClassificationEvaluationPipeline,  # type: ignore
            pruner=self.pruner_cls(n_warmup_steps=self.n_warmup_steps),
            sampler=self.sampler_cls(seed=self.seed),
            n_trials=self.n_trials,
            dataset_path=self.dataset_path,
            metric_name="f1__average_macro",
            metric_key="f1",
            config_space=self.config_space,
        )

    def _get_metadata(self, parameters: SampledParameters) -> FlairClassificationPipelineMetadata:
        (
            embedding_name,
            document_embedding_cls,
            task_train_kwargs,
            load_model_kwargs,
        ) = self._pop_sampled_parameters(parameters=parameters)
        metadata: FlairClassificationPipelineMetadata = {
            "embedding_name": embedding_name,
            "document_embedding_cls": document_embedding_cls,
            "dataset_name": str(self.dataset_name_or_path),
            "load_dataset_kwargs": self.load_dataset_kwargs,
            "input_column_name": self.input_column_name,
            "target_column_name": self.target_column_name,
            "task_model_kwargs": None,
            "task_train_kwargs": task_train_kwargs,
            "load_model_kwargs": load_model_kwargs,
        }
        return metadata

    def _get_evaluation_metadata(
        self, parameters: SampledParameters
    ) -> FlairClassificationEvaluationPipelineMetadata:
        (
            embedding_name,
            document_embedding_cls,
            task_train_kwargs,
            load_model_kwargs,
        ) = self._pop_sampled_parameters(parameters=parameters)
        metadata: FlairClassificationEvaluationPipelineMetadata = {
            "embedding_name": embedding_name,
            "document_embedding_cls": document_embedding_cls,
            "dataset_path": str(self.dataset_path),
            "persist_path": None,
            "predict_subset": "dev",
            "output_path": self.tmp_model_output_dir.name,
            "task_model_kwargs": None,
            "task_train_kwargs": task_train_kwargs,
            "load_model_kwargs": load_model_kwargs,
        }
        return metadata

    def _post_run_hook(self) -> None:
        super()._post_run_hook()
        self.tmp_dataset_dir.cleanup()
        self.tmp_model_output_dir.cleanup()


@dataclass
class OptimizedFlairPairClassificationPipeline(
    OptunaPipeline[
        FlairTextClassificationConfigSpace,
        FlairPairClassificationPipelineMetadata,
        FlairEvaluationPipelineMetadata,
        str,
        datasets.DatasetDict,
        Corpus,
    ],
    AbstractOptimizedFlairClassificationPipeline,
    _OptimizedFlairPairClassificationPipelineBase[FlairTextClassificationConfigSpace],
):
    def _init_preprocessing_pipeline(self) -> None:
        self.preprocessing_pipeline: Optional[FlairTextPairClassificationPreprocessingPipeline]
        if self.ignore_preprocessing_pipeline:
            self.preprocessing_pipeline = None
        else:
            self.preprocessing_pipeline = FlairTextPairClassificationPreprocessingPipeline(
                dataset_name=str(self.dataset_name_or_path),
                input_column_names=self.input_columns_names_pair,
                target_column_name=self.target_column_name,
                persist_path=str(self.dataset_path),
                sample_missing_splits=(self.sample_dev_split_fraction, None),
                ignore_test_subset=True,
                load_dataset_kwargs=self.load_dataset_kwargs,
            )

    def __post_init__(self) -> None:
        self._init_dataset_path()
        self._init_preprocessing_pipeline()
        super().__init__(
            preprocessing_pipeline=self.preprocessing_pipeline,
            evaluation_pipeline=FlairTextPairClassificationEvaluationPipeline,  # type: ignore
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
    ) -> FlairPairClassificationPipelineMetadata:
        (
            embedding_name,
            document_embedding_cls,
            task_train_kwargs,
            load_model_kwargs,
        ) = self._pop_sampled_parameters(parameters=parameters)
        metadata: FlairPairClassificationPipelineMetadata = {
            "embedding_name": embedding_name,
            "document_embedding_cls": document_embedding_cls,
            "dataset_name": str(self.dataset_name_or_path),
            "load_dataset_kwargs": self.load_dataset_kwargs,
            "input_columns_names_pair": self.input_columns_names_pair,
            "target_column_name": self.target_column_name,
            "task_model_kwargs": None,
            "task_train_kwargs": task_train_kwargs,
            "load_model_kwargs": load_model_kwargs,
        }
        return metadata

    def _get_evaluation_metadata(
        self, parameters: SampledParameters
    ) -> FlairClassificationEvaluationPipelineMetadata:
        (
            embedding_name,
            document_embedding_cls,
            task_train_kwargs,
            load_model_kwargs,
        ) = self._pop_sampled_parameters(parameters=parameters)
        metadata: FlairClassificationEvaluationPipelineMetadata = {
            "embedding_name": embedding_name,
            "dataset_path": str(self.dataset_path),
            "document_embedding_cls": document_embedding_cls,
            "task_model_kwargs": None,
            "task_train_kwargs": task_train_kwargs,
            "load_model_kwargs": load_model_kwargs,
            "persist_path": None,
            "predict_subset": "dev",
            "output_path": self.tmp_model_output_dir.name,
        }
        return metadata

    def _post_run_hook(self) -> None:
        super()._post_run_hook()
        self.tmp_dataset_dir.cleanup()
        self.tmp_model_output_dir.cleanup()


@dataclass
class OptimizedFlairSequenceLabelingPipeline(
    OptunaPipeline[
        FlairSequenceLabelingConfigSpace,
        FlairSequenceLabelingPipelineMetadata,
        FlairSequenceLabelingEvaluationPipelineMetadata,
        str,
        datasets.DatasetDict,
        Corpus,
    ],
    AbstractHuggingFaceOptimizedPipeline[FlairSequenceLabelingConfigSpace],
    _OptimizedFlairPipelineDefaultsBase,
    _OptimizedFlairPipelineBase[FlairSequenceLabelingConfigSpace],
):
    evaluation_mode: EvaluationMode = EvaluationMode.CONLL
    tagging_scheme: Optional[TaggingScheme] = None

    def _init_dataset_path(self) -> None:
        self.dataset_path: T_path
        if self.ignore_preprocessing_pipeline:
            self.dataset_path = Path(self.dataset_name_or_path)
            if not self.dataset_path.exists():
                raise FileNotFoundError("Dataset path not found")
        else:
            self.dataset_path = self.tmp_dataset_dir.name

    def _init_preprocessing_pipeline(self) -> None:
        self.preprocessing_pipeline: Optional[FlairSequenceLabelingPreprocessingPipeline]
        if self.ignore_preprocessing_pipeline:
            self.preprocessing_pipeline = None
        else:
            self.preprocessing_pipeline = FlairSequenceLabelingPreprocessingPipeline(
                dataset_name=str(self.dataset_name_or_path),
                input_column_name=self.input_column_name,
                target_column_name=self.target_column_name,
                persist_path=str(self.dataset_path),
                sample_missing_splits=(self.sample_dev_split_fraction, None),
                ignore_test_subset=True,
                load_dataset_kwargs=self.load_dataset_kwargs,
            )

    def __post_init__(self) -> None:
        self._init_dataset_path()
        self._init_preprocessing_pipeline()
        self.metric_name = SequenceLabelingEvaluator.get_metric_name(
            evaluation_mode=self.evaluation_mode, tagging_scheme=self.tagging_scheme
        )
        super().__init__(
            preprocessing_pipeline=self.preprocessing_pipeline,
            evaluation_pipeline=FlairSequenceLabelingEvaluationPipeline,  # type: ignore
            pruner=self.pruner_cls(n_warmup_steps=self.n_warmup_steps),
            sampler=self.sampler_cls(seed=self.seed),
            n_trials=self.n_trials,
            dataset_path=self.dataset_path,
            metric_name=self.metric_name,
            metric_key="overall_f1",
            config_space=self.config_space,
        )

    @staticmethod
    def _pop_sampled_parameters(
        parameters: SampledParameters,
    ) -> Tuple[str, int, Dict[str, ParameterValues], Dict[str, ParameterValues]]:
        embedding_name = parameters["embedding_name"]
        assert isinstance(embedding_name, str)
        hidden_size = parameters["hidden_size"]
        assert isinstance(hidden_size, int)
        task_train_kwargs = parameters["task_train_kwargs"]
        assert isinstance(task_train_kwargs, dict)
        task_model_kwargs = parameters["task_model_kwargs"]
        assert isinstance(task_model_kwargs, dict)
        return embedding_name, hidden_size, task_train_kwargs, task_model_kwargs

    def _get_metadata(self, parameters: SampledParameters) -> FlairSequenceLabelingPipelineMetadata:
        (
            embedding_name,
            hidden_size,
            task_train_kwargs,
            task_model_kwargs,
        ) = self._pop_sampled_parameters(parameters)
        metadata: FlairSequenceLabelingPipelineMetadata = {
            "embedding_name": embedding_name,
            "hidden_size": hidden_size,
            "dataset_name": str(self.dataset_name_or_path),
            "load_dataset_kwargs": self.load_dataset_kwargs,
            "input_column_name": self.input_column_name,
            "target_column_name": self.target_column_name,
            "evaluation_mode": self.evaluation_mode,
            "tagging_scheme": self.tagging_scheme,
            "task_train_kwargs": task_train_kwargs,
            "task_model_kwargs": task_model_kwargs,
        }
        return metadata

    def _get_evaluation_metadata(
        self, parameters: SampledParameters
    ) -> FlairSequenceLabelingEvaluationPipelineMetadata:
        (
            embedding_name,
            hidden_size,
            task_train_kwargs,
            task_model_kwargs,
        ) = self._pop_sampled_parameters(parameters)

        metadata: FlairSequenceLabelingEvaluationPipelineMetadata = {
            "embedding_name": embedding_name,
            "hidden_size": hidden_size,
            "dataset_path": str(self.dataset_path),
            "persist_path": None,
            "predict_subset": "dev",
            "output_path": self.tmp_model_output_dir.name,
            "evaluation_mode": self.evaluation_mode,
            "tagging_scheme": self.tagging_scheme,
            "task_train_kwargs": task_train_kwargs,
            "task_model_kwargs": task_model_kwargs,
        }
        return metadata

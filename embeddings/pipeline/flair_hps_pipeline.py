from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Generic, Optional, Tuple

from typing_extensions import Literal

from embeddings.hyperparameter_search.configspace import (
    ConfigSpace,
    SampledParameters,
    SequenceLabelingConfigSpace,
    TextClassificationConfigSpace,
)
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
    EvaluationPipelineMetadata,
    FlairClassificationEvaluationPipelineMetadata,
    FlairSequenceLabelingEvaluationPipelineMetadata,
    HuggingFaceClassificationPipelineMetadata,
    HuggingFacePairClassificationPipelineMetadata,
    HuggingFaceSequenceLabelingPipelineMetadata,
)
from embeddings.pipeline.preprocessing_pipeline import (
    FlairSequenceLabelingPreprocessingPipeline,
    FlairTextClassificationPreprocessingPipeline,
    FlairTextPairClassificationPreprocessingPipeline,
)
from embeddings.utils.utils import PrimitiveTypes


@dataclass
class _OptimizedFlairClassificationPipelineBase(
    _HuggingFaceOptimizedPipelineBase[ConfigSpace],
    ABC,
    Generic[ConfigSpace],
):
    input_column_name: str
    target_column_name: str


@dataclass
class _OptimizedFlairPairClassificationPipelineBase(
    _HuggingFaceOptimizedPipelineBase[ConfigSpace],
    ABC,
    Generic[ConfigSpace],
):
    input_columns_names_pair: Tuple[str, str]
    target_column_name: str


@dataclass
class _OptimizedFlairClassificationPipelineDefaultsBase(
    _HuggingFaceOptimizedPipelineDefaultsBase,
    ABC,
):
    dataset_dir: TemporaryDirectory[str] = field(init=False, default_factory=TemporaryDirectory)
    tmp_model_output_dir: TemporaryDirectory[str] = field(
        init=False, default_factory=TemporaryDirectory
    )

    @staticmethod
    def _pop_sampled_parameters(
        parameters: SampledParameters,
    ) -> Tuple[str, str, Dict[str, PrimitiveTypes], Dict[str, PrimitiveTypes]]:
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
        TextClassificationConfigSpace,
        HuggingFaceClassificationPipelineMetadata,
        EvaluationPipelineMetadata,
    ],
    AbstractHuggingFaceOptimizedPipeline[TextClassificationConfigSpace],
    _OptimizedFlairClassificationPipelineDefaultsBase,
    _OptimizedFlairClassificationPipelineBase[TextClassificationConfigSpace],
):
    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_dir.name).joinpath("ds.pkl")
        super().__init__(
            preprocessing_pipeline=FlairTextClassificationPreprocessingPipeline(
                dataset_name=self.dataset_name,
                input_column_name=self.input_column_name,
                target_column_name=self.target_column_name,
                persist_path=str(self.dataset_path),
                sample_missing_splits=(self.sample_dev_split_fraction, None),
                ignore_test_subset=True,
                load_dataset_kwargs=self.load_dataset_kwargs,
            ),
            evaluation_pipeline=FlairTextClassificationEvaluationPipeline,
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
    ) -> HuggingFaceClassificationPipelineMetadata:
        (
            embedding_name,
            document_embedding_cls,
            task_train_kwargs,
            load_model_kwargs,
        ) = self._pop_sampled_parameters(parameters=parameters)
        metadata: HuggingFaceClassificationPipelineMetadata = {
            "embedding_name": embedding_name,
            "document_embedding_cls": document_embedding_cls,
            "dataset_name": self.dataset_name,
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
        self.dataset_dir.cleanup()
        self.tmp_model_output_dir.cleanup()


@dataclass
class OptimizedFlairPairClassificationPipeline(
    OptunaPipeline[
        TextClassificationConfigSpace,
        HuggingFacePairClassificationPipelineMetadata,
        EvaluationPipelineMetadata,
    ],
    AbstractHuggingFaceOptimizedPipeline[TextClassificationConfigSpace],
    _OptimizedFlairClassificationPipelineDefaultsBase,
    _OptimizedFlairPairClassificationPipelineBase[TextClassificationConfigSpace],
):
    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_dir.name).joinpath("ds.pkl")
        super().__init__(
            preprocessing_pipeline=FlairTextPairClassificationPreprocessingPipeline(
                dataset_name=self.dataset_name,
                input_column_names=self.input_columns_names_pair,
                target_column_name=self.target_column_name,
                persist_path=str(self.dataset_path),
                sample_missing_splits=(self.sample_dev_split_fraction, None),
                ignore_test_subset=True,
                load_dataset_kwargs=self.load_dataset_kwargs,
            ),
            evaluation_pipeline=FlairTextPairClassificationEvaluationPipeline,
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
    ) -> HuggingFacePairClassificationPipelineMetadata:
        (
            embedding_name,
            document_embedding_cls,
            task_train_kwargs,
            load_model_kwargs,
        ) = self._pop_sampled_parameters(parameters=parameters)
        metadata: HuggingFacePairClassificationPipelineMetadata = {
            "embedding_name": embedding_name,
            "document_embedding_cls": document_embedding_cls,
            "dataset_name": self.dataset_name,
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
        self.dataset_dir.cleanup()
        self.tmp_model_output_dir.cleanup()


@dataclass
class _OptimizedFlairSequenceLabelingPipelineBase(
    _HuggingFaceOptimizedPipelineBase[ConfigSpace], ABC, Generic[ConfigSpace]
):
    input_column_name: str
    target_column_name: str


@dataclass
class OptimizedFlairSequenceLabelingPipeline(
    OptunaPipeline[
        SequenceLabelingConfigSpace,
        HuggingFaceSequenceLabelingPipelineMetadata,
        FlairSequenceLabelingEvaluationPipelineMetadata,
    ],
    AbstractHuggingFaceOptimizedPipeline[SequenceLabelingConfigSpace],
    _OptimizedFlairSequenceLabelingPipelineBase[SequenceLabelingConfigSpace],
):
    evaluation_mode: Literal["conll", "unit", "strict"] = "conll"
    tagging_scheme: Optional[str] = None
    dataset_path: TemporaryDirectory[str] = field(init=False, default_factory=TemporaryDirectory)
    tmp_model_output_dir: TemporaryDirectory[str] = field(
        init=False, default_factory=TemporaryDirectory
    )

    def _get_metric_name(self) -> str:
        if self.evaluation_mode == "unit":
            return "UnitSeqeval"
        elif self.evaluation_mode in {"conll", "strict"}:
            metric_name = "seqeval"
            if self.evaluation_mode == "conll":
                metric_name += "__mode_None"  # todo: deal with None in metric names
            else:
                metric_name += "__mode_strict"

            metric_name += f"__scheme_{self.tagging_scheme}"
            return metric_name
        else:
            raise ValueError(f"Evaluation Mode {self.evaluation_mode} unsupported.")

    def __post_init__(self) -> None:
        self.metric_name = self._get_metric_name()
        super().__init__(
            preprocessing_pipeline=FlairSequenceLabelingPreprocessingPipeline(
                dataset_name=self.dataset_name,
                input_column_name=self.input_column_name,
                target_column_name=self.target_column_name,
                persist_path=self.dataset_path.name,
                sample_missing_splits=(self.sample_dev_split_fraction, None),
                ignore_test_subset=True,
                load_dataset_kwargs=self.load_dataset_kwargs,
            ),
            config_space=self.config_space,
            evaluation_pipeline=FlairSequenceLabelingEvaluationPipeline,
            metric_name=self._get_metric_name(),
            metric_key="overall_f1",
            pruner=self.pruner_cls(n_warmup_steps=self.n_warmup_steps),
            sampler=self.sampler_cls(seed=self.seed),
            n_trials=self.n_trials,
            dataset_path=self.dataset_path,
        )

    def _pop_sampled_parameters(
        self,
        parameters: SampledParameters,
    ) -> Tuple[str, int, Dict[str, PrimitiveTypes], Dict[str, PrimitiveTypes]]:
        embedding_name = parameters["embedding_name"]
        assert isinstance(embedding_name, str)
        hidden_size = parameters["hidden_size"]
        assert isinstance(hidden_size, int)
        task_train_kwargs = parameters["task_train_kwargs"]
        assert isinstance(task_train_kwargs, dict)
        task_model_kwargs = parameters["task_model_kwargs"]
        assert isinstance(task_model_kwargs, dict)
        return embedding_name, hidden_size, task_train_kwargs, task_model_kwargs

    def _get_metadata(
        self, parameters: SampledParameters
    ) -> HuggingFaceSequenceLabelingPipelineMetadata:
        (
            embedding_name,
            hidden_size,
            task_train_kwargs,
            task_model_kwargs,
        ) = self._pop_sampled_parameters(parameters)
        metadata: HuggingFaceSequenceLabelingPipelineMetadata = {
            "embedding_name": embedding_name,
            "hidden_size": hidden_size,
            "dataset_name": self.dataset_name,
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
            "dataset_path": self.dataset_path.name,
            "persist_path": None,
            "predict_subset": "dev",
            "output_path": self.tmp_model_output_dir.name,
            "evaluation_mode": self.evaluation_mode,
            "tagging_scheme": self.tagging_scheme,
            "task_train_kwargs": task_train_kwargs,
            "task_model_kwargs": task_model_kwargs,
        }
        return metadata

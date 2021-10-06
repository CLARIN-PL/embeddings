import abc
import logging
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Generic, Literal, Optional, Tuple, Type, TypeVar, Union

import optuna
import pandas as pd
from optuna import Study

from embeddings.data.dataset import Data
from embeddings.data.io import T_path
from embeddings.hyperparameter_search.configspace import (
    ConfigSpace,
    FlairModelTrainerConfigSpace,
    SampledParameters,
    SequenceLabelingConfigSpace,
)
from embeddings.pipeline.evaluation_pipeline import (
    FlairSequenceLabelingEvaluationPipeline,
    FlairTextClassificationEvaluationPipeline,
    ModelEvaluationPipeline,
)
from embeddings.pipeline.pipelines_metadata import (
    EvaluationMetadata,
    EvaluationPipelineMetadata,
    FlairSequenceLabelingEvaluationPipelineMetadata,
    HuggingFaceClassificationPipelineMetadata,
    HuggingFaceSequenceLabelingPipelineMetadata,
    Metadata,
)
from embeddings.pipeline.preprocessing_pipeline import (
    FlairSequenceLabelingPreprocessingPipeline,
    FlairTextClassificationPreprocessingPipeline,
    PreprocessingPipeline,
)
from embeddings.pipeline.standard_pipeline import LoaderResult, ModelResult, TransformationResult
from embeddings.utils.hps_persister import HPSResultsPersister
from embeddings.utils.utils import PrimitiveTypes

EvaluationResult = TypeVar("EvaluationResult", bound=Dict[str, Dict[str, PrimitiveTypes]])


class OptimizedPipeline(ABC, Generic[Metadata]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def run(self) -> Tuple[pd.DataFrame, Metadata]:
        pass

    def persisting(
        self, best_params_path: T_path, log_path: T_path
    ) -> "PersistingPipeline[Metadata]":
        return PersistingPipeline(self, best_params_path, log_path)


class PersistingPipeline(OptimizedPipeline[Metadata]):
    def __init__(
        self, base_pipeline: OptimizedPipeline[Metadata], best_params_path: T_path, log_path: T_path
    ):
        self.base_pipeline = base_pipeline
        self.persister: HPSResultsPersister[Metadata] = HPSResultsPersister(
            best_params_path=best_params_path, log_path=log_path
        )

    def run(self) -> Tuple[pd.DataFrame, Metadata]:
        result = self.base_pipeline.run()
        self.persister.persist(result)
        return result


class OptunaPipeline(
    OptimizedPipeline[Metadata], Generic[ConfigSpace, Metadata, EvaluationMetadata]
):
    def __init__(
        self,
        config_space: ConfigSpace,
        preprocessing_pipeline: PreprocessingPipeline[Data, LoaderResult, TransformationResult],
        evaluation_pipeline: Type[
            ModelEvaluationPipeline[Data, LoaderResult, ModelResult, EvaluationResult]
        ],
        pruner: optuna.pruners.BasePruner,
        sampler: optuna.samplers.BaseSampler,
        n_trials: int,
        dataset_path: Union[str, Path, "TemporaryDirectory[str]"],
        metric_name: str,
        metric_key: str,
    ):
        self.config_space = config_space
        self.preprocessing_pipeline = preprocessing_pipeline
        self.evaluation_pipeline = evaluation_pipeline
        self.pruner = pruner
        self.sampler = sampler
        self.n_trials = n_trials
        self.dataset_path = dataset_path
        self.metric_name = metric_name
        self.metric_key = metric_key

    @abc.abstractmethod
    def _get_metadata(self, parameters: SampledParameters) -> Metadata:
        pass

    @abc.abstractmethod
    def _get_evaluation_metadata(self, parameters: SampledParameters) -> EvaluationMetadata:
        pass

    def get_best_paramaters(self, study: Study) -> Metadata:
        best_params = study.best_params
        parsed_params = self.config_space.parse_parameters(best_params)
        return self._get_metadata(parsed_params)

    def run(
        self,
    ) -> Tuple[pd.DataFrame, Metadata]:
        self._pre_run_hook()
        self.preprocessing_pipeline.run()
        study: Study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
        )
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        if isinstance(self.dataset_path, TemporaryDirectory):
            self.dataset_path.cleanup()

        metadata: Metadata = self.get_best_paramaters(study)
        self._post_run_hook()
        return study.trials_dataframe(), metadata

    def objective(self, trial: optuna.trial.Trial) -> float:
        parameters = self.config_space.sample_parameters(trial=trial)
        parsed_params = self.config_space.parse_parameters(parameters)
        args = self._get_evaluation_metadata(parsed_params)

        pipeline = self.evaluation_pipeline(**args)
        results = pipeline.run()
        metric = results[self.metric_name][self.metric_key]
        assert isinstance(metric, float)
        return metric

    def _pre_run_hook(self) -> None:
        logging.getLogger("flair").setLevel(logging.WARNING)

    def _post_run_hook(self) -> None:
        logging.getLogger("flair").setLevel(logging.INFO)


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class BaseHuggingFaceOptimizedPipeline(ABC):
    dataset_name: str
    input_column_name: str
    target_column_name: str
    n_warmup_steps: int = 10
    n_trials: int = 2
    sample_dev_split_fraction: Optional[float] = 0.1
    seed: int = 441
    fine_tune_embeddings: bool = False
    pruner_cls: Type[optuna.pruners.MedianPruner] = field(
        init=False, default=optuna.pruners.MedianPruner
    )
    sampler_cls: Type[optuna.samplers.TPESampler] = field(
        init=False, default=optuna.samplers.TPESampler
    )

    @abc.abstractmethod
    def __post_init__(self) -> None:
        pass


@dataclass
class OptimizedFlairClassificationPipeline(
    OptunaPipeline[
        FlairModelTrainerConfigSpace,
        HuggingFaceClassificationPipelineMetadata,
        EvaluationPipelineMetadata,
    ],
    BaseHuggingFaceOptimizedPipeline,
):
    dataset_dir: TemporaryDirectory[str] = field(init=False, default_factory=TemporaryDirectory)
    tmp_model_output_dir: TemporaryDirectory[str] = field(
        init=False, default_factory=TemporaryDirectory
    )
    config_space: FlairModelTrainerConfigSpace = FlairModelTrainerConfigSpace()

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
        task_train_kwargs = parameters["task_train_kwargs"]
        assert isinstance(task_train_kwargs, dict)
        embedding_name = parameters["embedding_name"]
        assert isinstance(embedding_name, str)
        metadata: HuggingFaceClassificationPipelineMetadata = {
            "embedding_name": embedding_name,
            "dataset_name": self.dataset_name,
            "input_column_name": self.input_column_name,
            "target_column_name": self.target_column_name,
            "task_model_kwargs": None,
            "task_train_kwargs": task_train_kwargs,
        }
        return metadata

    def _get_evaluation_metadata(self, parameters: SampledParameters) -> EvaluationPipelineMetadata:
        task_train_kwargs = parameters["task_train_kwargs"]
        assert isinstance(task_train_kwargs, dict)
        embedding_name = parameters["embedding_name"]
        assert isinstance(embedding_name, str)
        metadata: EvaluationPipelineMetadata = {
            "embedding_name": embedding_name,
            "dataset_path": str(self.dataset_path),
            "task_model_kwargs": None,
            "task_train_kwargs": task_train_kwargs,
            "persist_path": None,
            "predict_subset": "dev",
            "fine_tune_embeddings": False,
            "output_path": self.tmp_model_output_dir.name,
        }
        return metadata

    def _post_run_hook(self) -> None:
        super()._post_run_hook()
        self.dataset_dir.cleanup()
        self.tmp_model_output_dir.cleanup()


@dataclass
class OptimizedFlairSequenceLabelingPipeline(
    OptunaPipeline[
        SequenceLabelingConfigSpace,
        HuggingFaceSequenceLabelingPipelineMetadata,
        FlairSequenceLabelingEvaluationPipelineMetadata,
    ],
    BaseHuggingFaceOptimizedPipeline,
):
    evaluation_mode: Literal["conll", "unit", "strict"] = "conll"
    tagging_scheme: Optional[str] = None
    config_space: SequenceLabelingConfigSpace = SequenceLabelingConfigSpace()
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

    def _get_metadata(
        self, parameters: SampledParameters
    ) -> HuggingFaceSequenceLabelingPipelineMetadata:
        task_train_kwargs = parameters["task_train_kwargs"]
        assert isinstance(task_train_kwargs, dict)
        task_model_kwargs = parameters["task_model_kwargs"]
        assert isinstance(task_model_kwargs, dict)
        hidden_size = parameters["hidden_size"]
        assert isinstance(hidden_size, int)
        embedding_name = parameters["embedding_name"]
        assert isinstance(embedding_name, str)

        metadata: HuggingFaceSequenceLabelingPipelineMetadata = {
            "embedding_name": embedding_name,
            "dataset_name": self.dataset_name,
            "input_column_name": self.input_column_name,
            "target_column_name": self.target_column_name,
            "evaluation_mode": self.evaluation_mode,
            # "fine_tune_embeddings": self.fine_tune_embeddings,
            "hidden_size": hidden_size,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
            "tagging_scheme": self.tagging_scheme,
        }
        return metadata

    def _get_evaluation_metadata(
        self, parameters: SampledParameters
    ) -> FlairSequenceLabelingEvaluationPipelineMetadata:
        task_train_kwargs = parameters["task_train_kwargs"]
        assert isinstance(task_train_kwargs, dict)
        task_model_kwargs = parameters["task_model_kwargs"]
        assert isinstance(task_model_kwargs, dict)
        hidden_size = parameters["hidden_size"]
        assert isinstance(hidden_size, int)
        embedding_name = parameters["embedding_name"]
        assert isinstance(embedding_name, str)

        metadata: FlairSequenceLabelingEvaluationPipelineMetadata = {
            "embedding_name": embedding_name,
            "dataset_path": self.dataset_path.name,
            "task_model_kwargs": None,
            "task_train_kwargs": task_train_kwargs,
            "persist_path": None,
            "predict_subset": "dev",
            "fine_tune_embeddings": False,
            "output_path": self.tmp_model_output_dir.name,
            "hidden_size": hidden_size,
            "evaluation_mode": self.evaluation_mode,
            "tagging_scheme": self.tagging_scheme,
        }
        return metadata

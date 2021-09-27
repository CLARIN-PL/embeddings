import abc
import logging
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generic, Optional, Tuple, Type, Union

import optuna
import pandas as pd
from optuna import Study

from embeddings.data.dataset import Data
from embeddings.data.io import T_path
from embeddings.hyperparameter_search.configspace import (
    FlairModelTrainerConfigSpace,
    SequenceLabelingConfigSpace,
)
from embeddings.pipeline.evaluation_pipeline import (
    FlairSequenceLabelingEvaluationPipeline,
    FlairTextClassificationEvaluationPipeline,
)
from embeddings.pipeline.pipelines_metadata import (
    HuggingFaceClassificationPipelineMetadata,
    HuggingFaceSequenceLabelingPipelineMetadata,
    Metadata,
)
from embeddings.pipeline.preprocessing_pipeline import (
    FlairSequenceLabelingPreprocessingPipeline,
    FlairTextClassificationPreprocessingPipeline,
    PreprocessingPipeline,
)
from embeddings.pipeline.standard_pipeline import LoaderResult, TransformationResult
from embeddings.utils.hps_persister import HPSResultsPersister


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


class OptunaPipeline(OptimizedPipeline[Metadata]):
    def __init__(
        self,
        preprocessing_pipeline: PreprocessingPipeline[Data, LoaderResult, TransformationResult],
        pruner: optuna.pruners.BasePruner,
        sampler: optuna.samplers.BaseSampler,
        n_trials: int,
        dataset_path: Union[str, Path, "TemporaryDirectory[str]"],
    ):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.pruner: optuna.pruners.BasePruner = pruner
        self.sampler: optuna.samplers.BaseSampler = sampler
        self.n_trials: int = n_trials
        self.dataset_path = dataset_path

    def _pre_run_hook(self) -> None:
        pass

    def _post_run_hook(self) -> None:
        pass

    @abc.abstractmethod
    def objective(self, trial: optuna.trial.Trial) -> float:
        pass

    @abc.abstractmethod
    def _get_best_params_metadata(self, study: Study) -> Metadata:
        pass

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

        metadata: Metadata = self._get_best_params_metadata(study)
        self._post_run_hook()
        return study.trials_dataframe(), metadata


class FlairOptimizedPipeline(OptunaPipeline[Metadata], ABC):
    def _pre_run_hook(self) -> None:
        logging.getLogger("flair").setLevel(logging.WARNING)

    def _post_run_hook(self) -> None:
        logging.getLogger("flair").setLevel(logging.INFO)


@dataclass  # type: ignore
class HuggingFaceOptimizedPipeline(OptimizedPipeline[Metadata], ABC):
    dataset_name: str
    embedding_name: str
    input_column_name: str
    target_column_name: str
    n_warmup_steps: int = 10
    n_trials: int = 2
    seed: int = 441
    fine_tune_embeddings: bool = False
    pruner_cls: Type[optuna.pruners.MedianPruner] = field(
        init=False, default=optuna.pruners.MedianPruner
    )
    sampler_cls: Type[optuna.samplers.TPESampler] = field(
        init=False, default=optuna.samplers.TPESampler
    )


@dataclass
class OptimizedFlairClassificationPipeline(
    FlairOptimizedPipeline[HuggingFaceClassificationPipelineMetadata],
    HuggingFaceOptimizedPipeline[HuggingFaceClassificationPipelineMetadata],
):
    dataset_dir: TemporaryDirectory[str] = field(init=False, default=TemporaryDirectory())
    config_space: FlairModelTrainerConfigSpace = FlairModelTrainerConfigSpace()

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_dir.name).joinpath("ds.pkl")
        super().__init__(
            preprocessing_pipeline=FlairTextClassificationPreprocessingPipeline(
                dataset_name=self.dataset_name,
                input_column_name=self.input_column_name,
                target_column_name=self.target_column_name,
                persist_path=str(self.dataset_path),
                sample_missing_splits=True,
                ignore_test_subset=True,
            ),
            pruner=self.pruner_cls(n_warmup_steps=self.n_warmup_steps),
            sampler=self.sampler_cls(seed=self.seed),
            n_trials=self.n_trials,
            dataset_path=self.dataset_path,
        )

    def _post_run_hook(self) -> None:
        super()._post_run_hook()
        self.dataset_dir.cleanup()

    def objective(self, trial: optuna.trial.Trial) -> float:
        parameters = self.config_space.sample_parameters(trial=trial)
        task_train_kwargs = self.config_space.parse_parameters(parameters)

        tmp_dir = TemporaryDirectory()
        pipeline = FlairTextClassificationEvaluationPipeline(
            dataset_path=str(self.dataset_path),
            embedding_name=self.embedding_name,
            task_model_kwargs=None,
            task_train_kwargs=task_train_kwargs,
            output_path=tmp_dir.name,
            predict_subset="dev",
            fine_tune_embeddings=self.fine_tune_embeddings,
        )
        results = pipeline.run()
        metric = results["f1__average_macro"]["f1"]
        assert isinstance(metric, float)
        return metric

    def _get_best_params_metadata(self, study: Study) -> HuggingFaceClassificationPipelineMetadata:
        best_params = study.best_params
        task_train_kwargs = self.config_space.parse_parameters(best_params)

        metadata: HuggingFaceClassificationPipelineMetadata = {
            "embedding_name": self.embedding_name,
            "dataset_name": self.dataset_name,
            "input_column_name": self.input_column_name,
            "target_column_name": self.target_column_name,
            # "fine_tune_embeddings": self.fine_tune_embeddings,
            "task_model_kwargs": None,
            "task_train_kwargs": task_train_kwargs,
        }
        return metadata


@dataclass
class OptimizedFlairSequenceLabelingPipeline(
    FlairOptimizedPipeline[HuggingFaceSequenceLabelingPipelineMetadata],
    HuggingFaceOptimizedPipeline[HuggingFaceSequenceLabelingPipelineMetadata],
):
    evaluation_mode: str = "conll"
    tagging_scheme: Optional[str] = None
    config_space: SequenceLabelingConfigSpace = SequenceLabelingConfigSpace()
    dataset_path: "TemporaryDirectory[str]" = field(init=False, default=TemporaryDirectory())

    def __post_init__(self) -> None:
        super().__init__(
            preprocessing_pipeline=FlairSequenceLabelingPreprocessingPipeline(
                dataset_name=self.dataset_name,
                input_column_name=self.input_column_name,
                target_column_name=self.target_column_name,
                persist_path=self.dataset_path.name,
                sample_missing_splits=True,
                ignore_test_subset=True,
            ),
            pruner=self.pruner_cls(n_warmup_steps=self.n_warmup_steps),
            sampler=self.sampler_cls(seed=self.seed),
            n_trials=self.n_trials,
            dataset_path=self.dataset_path,
        )

    def objective(self, trial: optuna.trial.Trial) -> float:
        parameters = self.config_space.sample_parameters(trial=trial)
        hidden_size, task_model_kwargs, task_train_kwargs = self.config_space.parse_parameters(
            parameters
        )

        tmp_dir = TemporaryDirectory()
        pipeline = FlairSequenceLabelingEvaluationPipeline(
            dataset_path=self.dataset_path.name,
            hidden_size=hidden_size,
            embedding_name=self.embedding_name,
            evaluation_mode=self.evaluation_mode,
            fine_tune_embeddings=self.fine_tune_embeddings,
            task_model_kwargs=task_model_kwargs,
            task_train_kwargs=task_train_kwargs,
            output_path=tmp_dir.name,
            predict_subset="dev",
        )
        results = pipeline.run()
        metric = results["seqeval__mode_None__scheme_None"]["overall_f1"]
        assert isinstance(metric, float)
        return metric

    def _get_best_params_metadata(
        self, study: Study
    ) -> HuggingFaceSequenceLabelingPipelineMetadata:
        best_params = study.best_params
        hidden_size, task_model_kwargs, task_train_kwargs = self.config_space.parse_parameters(
            best_params
        )

        metadata: HuggingFaceSequenceLabelingPipelineMetadata = {
            "embedding_name": self.embedding_name,
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

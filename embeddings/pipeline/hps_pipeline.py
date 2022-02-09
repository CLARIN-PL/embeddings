import abc
import logging
from abc import ABC
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar, Union

import optuna
import pandas as pd
from optuna import Study

from embeddings.data.dataset import Data
from embeddings.data.io import T_path
from embeddings.hyperparameter_search.configspace import ConfigSpace, SampledParameters
from embeddings.pipeline.evaluation_pipeline import ModelEvaluationPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.pipeline.pipelines_metadata import EvaluationMetadata, Metadata
from embeddings.pipeline.preprocessing_pipeline import PreprocessingPipeline
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
    OptimizedPipeline[Metadata],
    Generic[ConfigSpace, Metadata, EvaluationMetadata, Data, LoaderResult, TransformationResult],
):
    def __init__(
        self,
        config_space: ConfigSpace,
        preprocessing_pipeline: Optional[
            PreprocessingPipeline[Data, LoaderResult, TransformationResult]
        ],
        evaluation_pipeline: Union[
            Type[ModelEvaluationPipeline[Data, LoaderResult, ModelResult, EvaluationResult]],
            Type[LightningPipeline[Data, ModelResult, EvaluationResult]],
        ],
        pruner: optuna.pruners.BasePruner,
        sampler: optuna.samplers.BaseSampler,
        n_trials: int,
        dataset_path: T_path,
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
        constant_params = study.best_trial.user_attrs
        parsed_params = self.config_space.parse_parameters(best_params | constant_params)
        return self._get_metadata(parsed_params)

    def run(
        self,
    ) -> Tuple[pd.DataFrame, Metadata]:
        self._pre_run_hook()
        if self.preprocessing_pipeline is not None:
            self.preprocessing_pipeline.run()
        study: Study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
        )
        study.optimize(
            self.objective, n_trials=self.n_trials, show_progress_bar=True, catch=(Exception,)
        )

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
        logging.getLogger("optuna").setLevel(logging.WARNING)

    def _post_run_hook(self) -> None:
        logging.getLogger("optuna").setLevel(logging.INFO)


@dataclass
class _HuggingFaceOptimizedPipelineBase(ABC, Generic[ConfigSpace]):
    dataset_name_or_path: T_path
    config_space: ConfigSpace


@dataclass
class _HuggingFaceOptimizedPipelineDefaultsBase(ABC):
    load_dataset_kwargs: Optional[Dict[str, Any]] = None
    n_warmup_steps: int = 10
    n_trials: int = 2
    sample_dev_split_fraction: Optional[float] = 0.1
    seed: int = 441
    pruner_cls: Type[optuna.pruners.MedianPruner] = field(
        init=False, default=optuna.pruners.MedianPruner
    )
    sampler_cls: Type[optuna.samplers.TPESampler] = field(
        init=False, default=optuna.samplers.TPESampler
    )
    ignore_preprocessing_pipeline: bool = False


# Mypy currently properly don't handle dataclasses with abstract methods  https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class AbstractHuggingFaceOptimizedPipeline(
    _HuggingFaceOptimizedPipelineDefaultsBase,
    _HuggingFaceOptimizedPipelineBase[ConfigSpace],
    ABC,
    Generic[ConfigSpace],
):
    @abc.abstractmethod
    def __post_init__(self) -> None:
        pass

    @abc.abstractmethod
    def _init_dataset_path(self) -> None:
        pass

    @abc.abstractmethod
    def _init_preprocessing_pipeline(self) -> None:
        pass

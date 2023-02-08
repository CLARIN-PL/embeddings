import abc
import logging
import os
from abc import ABC
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generic, Optional, Tuple, Type, TypeVar

import optuna
import pandas as pd
from optuna import Study

from embeddings.config.config_space import ConfigSpace, SampledParameters
from embeddings.data.dataset import Data
from embeddings.data.io import T_path
from embeddings.evaluator.evaluation_results import EvaluationResults
from embeddings.pipeline.lightning_pipeline import LightningPipeline
from embeddings.pipeline.pipelines_metadata import EvaluationMetadata, Metadata
from embeddings.pipeline.preprocessing_pipeline import PreprocessingPipeline
from embeddings.pipeline.standard_pipeline import LoaderResult, ModelResult, TransformationResult
from embeddings.utils.hps_persister import HPSResultsPersister
from embeddings.utils.loggers import LightningLoggingConfig
from embeddings.utils.torch_utils import cleanup_torch_model_artifacts
from embeddings.utils.utils import standardize_name

EvaluationResult = TypeVar("EvaluationResult", bound=EvaluationResults)


class OptunaCallback(ABC):
    @abc.abstractmethod
    def __call__(self, study: Study, trial: optuna.trial.FrozenTrial) -> None:
        pass


class TorchGarbageCollectorOptunaCallback(OptunaCallback):
    def __call__(self, study: Study, trial: optuna.trial.FrozenTrial) -> None:
        cleanup_torch_model_artifacts()


class OptimizedPipeline(ABC, Generic[Metadata]):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def run(self, **kwargs: Any) -> Tuple[pd.DataFrame, Metadata]:
        pass

    def persisting(
        self,
        best_params_path: T_path,
        log_path: T_path,
        logging_config: LightningLoggingConfig = LightningLoggingConfig(),
        logging_hps_summary_name: Optional[str] = None,
    ) -> "PersistingPipeline[Metadata]":
        return PersistingPipeline(
            self, best_params_path, log_path, logging_config, logging_hps_summary_name
        )


class PersistingPipeline(OptimizedPipeline[Metadata]):
    def __init__(
        self,
        base_pipeline: OptimizedPipeline[Metadata],
        best_params_path: T_path,
        log_path: T_path,
        logging_config: LightningLoggingConfig = LightningLoggingConfig(),
        logging_hps_summary_name: Optional[str] = None,
    ):
        self.base_pipeline = base_pipeline
        self.persister: HPSResultsPersister[Metadata] = HPSResultsPersister(
            best_params_path=best_params_path,
            log_path=log_path,
            logging_config=logging_config,
            logging_hps_summary_name=logging_hps_summary_name,
        )

    def run(self, **kwargs: Any) -> Tuple[pd.DataFrame, Metadata]:
        result = self.base_pipeline.run(**kwargs)
        self.persister.persist(result)
        return result


class OptunaPipeline(
    OptimizedPipeline[Metadata],
    Generic[
        ConfigSpace,
        Metadata,
        EvaluationMetadata,
        Data,
        LoaderResult,
        TransformationResult,
        ModelResult,
        EvaluationResult,
    ],
):
    def __init__(
        self,
        config_space: ConfigSpace,
        preprocessing_pipeline: Optional[
            PreprocessingPipeline[Data, LoaderResult, TransformationResult]
        ],
        evaluation_pipeline: Type[
            LightningPipeline[TransformationResult, ModelResult, EvaluationResult]
        ],
        pruner: optuna.pruners.BasePruner,
        sampler: optuna.samplers.BaseSampler,
        n_trials: int,
        dataset_path: T_path,
        metric_name: str,
    ):
        self.config_space: ConfigSpace = config_space
        self.preprocessing_pipeline = preprocessing_pipeline
        self.evaluation_pipeline = evaluation_pipeline
        self.pruner = pruner
        self.sampler = sampler
        self.n_trials = n_trials
        self.dataset_path = dataset_path
        self.metric_name = metric_name

    @abc.abstractmethod
    def _get_metadata(self, parameters: SampledParameters) -> Metadata:
        pass

    @abc.abstractmethod
    def _get_evaluation_metadata(
        self, parameters: SampledParameters, **kwargs: Any
    ) -> EvaluationMetadata:
        pass

    def get_best_paramaters(self, study: Study) -> Metadata:
        best_params = study.best_params
        constant_params = study.best_trial.user_attrs
        parsed_params = self.config_space.parse_parameters(dict(best_params, **constant_params))
        return self._get_metadata(parsed_params)

    def run(
        self,
        run_name: Optional[str] = None,
        catch: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs: Any,
    ) -> Tuple[pd.DataFrame, Metadata]:
        self._pre_run_hook()
        if self.preprocessing_pipeline is not None:
            self.preprocessing_pipeline.run()
        study: Study = optuna.create_study(
            direction="maximize", sampler=self.sampler, pruner=self.pruner, study_name=run_name
        )
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=catch,
            callbacks=[TorchGarbageCollectorOptunaCallback()],
        )

        if isinstance(self.dataset_path, TemporaryDirectory):
            self.dataset_path.cleanup()

        metadata: Metadata = self.get_best_paramaters(study)
        self._post_run_hook()
        return study.trials_dataframe(), metadata

    def objective(self, trial: optuna.trial.Trial) -> float:
        trial_name = standardize_name(f"study_{trial.study.study_name}_trial_{trial.number}")
        parameters = self.config_space.sample_parameters(trial=trial)
        parsed_params = self.config_space.parse_parameters(parameters)
        kwargs = self._get_evaluation_metadata(parsed_params, trial_name=trial_name)
        os.makedirs(kwargs["output_path"], exist_ok=True)
        pipeline = self._get_evaluation_pipeline(**kwargs)
        results = pipeline.run(run_name=trial_name)
        metric = getattr(results, self.metric_name)
        assert isinstance(metric, float)
        return metric

    def _get_evaluation_pipeline(
        self, **kwargs: Any
    ) -> LightningPipeline[TransformationResult, ModelResult, EvaluationResult]:
        return self.evaluation_pipeline(**kwargs)

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


@dataclass
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

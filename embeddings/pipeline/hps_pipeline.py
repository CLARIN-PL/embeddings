import logging
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Dict, Tuple, Type, Union

import datasets
import optuna
import pandas as pd
from flair.data import Corpus
from optuna import Study

from embeddings.hyperparameter_search.configspace import SequenceLabelingConfigSpace
from embeddings.pipeline.evaluation_pipeline import FlairSequenceLabelingEvaluationPipeline
from embeddings.pipeline.preprocessing_pipeline import (
    FlairSequenceLabelingPreprocessingPipeline,
    PreprocessingPipeline,
)
from embeddings.utils.utils import PrimitiveTypes


@dataclass
class OptimizedFlairSequenceLabelingPipeline:
    dataset_name: str
    input_column_name: str
    target_column_name: str
    embedding_name: str
    fine_tune_embeddings: bool = False
    evaluation_mode: str = "conll"
    config_space: SequenceLabelingConfigSpace = SequenceLabelingConfigSpace()
    seed: int = 441
    n_warmup_steps: int = 10
    n_trials: int = 20
    pruner: Type[optuna.pruners.MedianPruner] = field(
        init=False, default=optuna.pruners.MedianPruner
    )
    sampler: Type[optuna.samplers.TPESampler] = field(
        init=False, default=optuna.samplers.TPESampler
    )
    dataset_path: "TemporaryDirectory[str]" = field(init=False, default=TemporaryDirectory())

    def __post_init__(self) -> None:
        logging.getLogger("flair").setLevel(logging.WARNING)

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
    ) -> Dict[str, Union[PrimitiveTypes, Dict[str, PrimitiveTypes]]]:
        best_params = study.best_params
        hidden_size, task_model_kwargs, task_train_kwargs = self.config_space.parse_parameters(
            best_params
        )
        return {
            "dataset_name": self.dataset_name,
            "input_column_name": self.input_column_name,
            "target_column_name": self.target_column_name,
            "evaluation_mode": self.evaluation_mode,
            "fine_tune_embeddings": self.fine_tune_embeddings,
            "hidden_size": hidden_size,
            "task_model_kwargs": task_model_kwargs,
            "task_train_kwargs": task_train_kwargs,
        }

    def run(
        self,
    ) -> Tuple[pd.DataFrame, Dict[str, Union[PrimitiveTypes, Dict[str, PrimitiveTypes]]]]:
        preprocessing_pipeline: PreprocessingPipeline[
            str, datasets.DatasetDict, Corpus
        ] = FlairSequenceLabelingPreprocessingPipeline(
            dataset_name=self.dataset_name,
            input_column_name=self.input_column_name,
            target_column_name=self.target_column_name,
            persist_path=self.dataset_path.name,
            sample_missing_splits=True,
            ignore_test_subset=True,
        )
        preprocessing_pipeline.run()
        study: Study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler(seed=self.seed),
            pruner=self.pruner(n_warmup_steps=self.n_warmup_steps),
        )
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        self.dataset_path.cleanup()
        metadata = self._get_best_params_metadata(study)
        return study.trials_dataframe(), metadata

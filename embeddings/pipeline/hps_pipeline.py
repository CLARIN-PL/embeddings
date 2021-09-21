from dataclasses import dataclass
from tempfile import TemporaryDirectory

import datasets
import optuna
import pandas as pd
from flair.data import Corpus

from embeddings.hyperparameter_search.configspace import SequenceLabelingConfigSpace
from embeddings.pipeline.evaluation_pipeline import FlairSequenceLabelingEvaluationPipeline
from embeddings.pipeline.preprocessing_pipeline import (
    FlairSequenceLabelingPreprocessingPipeline,
    PreprocessingPipeline,
)


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
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner
    sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler
    n_trials: int = 20

    def objective(self, trial: optuna.trial.Trial, dataset_path: str) -> float:
        parameters = self.config_space.sample_parameters(trial=trial)
        hidden_size, task_model_kwargs, task_train_kwargs = self.config_space.parse_parameters(
            parameters
        )

        tmp_dir = TemporaryDirectory()
        pipeline = FlairSequenceLabelingEvaluationPipeline(
            dataset_path=dataset_path,
            hidden_size=hidden_size,
            embedding_name=self.embedding_name,
            evaluation_mode=self.evaluation_mode,
            fine_tune_embeddings=self.fine_tune_embeddings,
            task_model_kwargs=task_model_kwargs,
            task_train_kwargs=task_train_kwargs,
            output_path=tmp_dir.name,
        )
        results = pipeline.run()
        metric = results["seqeval__mode_None__scheme_None"]["overall_f1"]
        assert isinstance(metric, float)
        return metric

    def run(self) -> pd.DataFrame:
        dataset_path: TemporaryDirectory[str] = TemporaryDirectory()
        preprocessing_pipeline: PreprocessingPipeline[
            str, datasets.DatasetDict, Corpus
        ] = FlairSequenceLabelingPreprocessingPipeline(
            dataset_name=self.dataset_name,
            input_column_name=self.input_column_name,
            target_column_name=self.target_column_name,
            persist_path=dataset_path.name,
        )
        preprocessing_pipeline.run()
        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler(seed=self.seed),
            pruner=self.pruner(n_warmup_steps=self.n_warmup_steps),
        )
        study.optimize(self.objective, n_trials=self.n_trials)
        dataset_path.cleanup()
        return study.trials_dataframe()

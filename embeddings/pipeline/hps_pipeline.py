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


class OptimizedFlairSequenceLabelingPipeline:
    def __init__(
        self,
        dataset_name: str,
        input_column_name: str,
        target_column_name: str,
        embedding_name: str,
        fine_tune_embeddings: bool = False,
        evaluation_mode: str = "conll",
        config_space: SequenceLabelingConfigSpace = SequenceLabelingConfigSpace(),
    ) -> None:
        self.dataset_name = dataset_name
        self.input_column_name = input_column_name
        self.target_column_name = target_column_name
        self.config_space: SequenceLabelingConfigSpace = config_space
        self.embedding_name = embedding_name
        self.fine_tune_embeddings = fine_tune_embeddings
        self.evaluation_mode = evaluation_mode

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
            sampler=optuna.samplers.TPESampler(seed=441),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        study.optimize(self.objective, n_trials=20)
        dataset_path.cleanup()
        return study.trials_dataframe()

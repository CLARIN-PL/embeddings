from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar

import srsly
import yaml

from embeddings.data.datamodule import BaseDataModule, Data
from embeddings.evaluator.evaluator import Evaluator
from embeddings.model.model import Model
from embeddings.pipeline.pipeline import Pipeline
from embeddings.utils.loggers import LightningLoggingConfig, LightningWandbWrapper
from embeddings.utils.utils import get_installed_packages, standardize_name

EvaluationResult = TypeVar("EvaluationResult")
ModelResult = TypeVar("ModelResult")


class LightningPipeline(
    Pipeline[EvaluationResult],
    Generic[Data, ModelResult, EvaluationResult],
):
    DEFAULT_TASK_TRAIN_KWARGS: Dict[str, Any] = {"devices": "auto", "accelerator": "auto"}
    DEFAULT_TASK_MODEL_KWARGS: Dict[str, Any] = {"use_scheduler": True}
    DEFAULT_DATAMODULE_KWARGS: Dict[str, Any] = {"max_seq_length": None}
    DEFAULT_MODEL_CONFIG_KWARGS: Dict[str, Any] = {"classifier_dropout": None}
    DEFAULT_EARLY_STOPPING_KWARGS: Dict[str, Any] = {
        "monitor": "val/Loss",
        "mode": "min",
        "patience": 3,
    }

    def __init__(
        self,
        datamodule: BaseDataModule[Data],
        model: Model[BaseDataModule[Data], ModelResult],
        evaluator: Evaluator[ModelResult, EvaluationResult],
        output_path: Path,
        logging_config: LightningLoggingConfig,
        pipeline_kwargs: Dict[str, Any],
    ) -> None:
        self.datamodule = datamodule
        self.model = model
        self.evaluator = evaluator
        self.output_path = output_path
        self.logging_config = logging_config
        self.pipeline_kwargs = pipeline_kwargs
        self.pipeline_kwargs.pop("self")
        self.pipeline_kwargs.pop("pipeline_kwargs")
        self.result: Optional[EvaluationResult] = None

    def run(self, run_name: Optional[str] = None) -> EvaluationResult:
        if run_name:
            run_name = standardize_name(run_name)
        self._save_artifacts()
        model_result = self.model.execute(data=self.datamodule, run_name=run_name)
        self.result = self.evaluator.evaluate(model_result)
        self._save_metrics()
        self._finish_logging()
        return self.result

    def _save_artifacts(self) -> None:
        srsly.write_json(self.output_path / "packages.json", get_installed_packages())
        with open(self.output_path / "pipeline_config.yaml", "w") as f:
            yaml.dump(self.pipeline_kwargs, stream=f)

    def _save_metrics(self) -> None:
        metrics = getattr(self.result, "metrics")
        with open(self.output_path / "metrics.yaml", "w") as f:
            yaml.dump(metrics, stream=f)

    def _finish_logging(self) -> None:
        if self.logging_config.use_wandb():
            wrapper = LightningWandbWrapper(self.logging_config)
            wrapper.log_output(
                self.output_path, ignore={"wandb", "csv", "tensorboard", "checkpoints"}
            )
            metrics = getattr(self.result, "metrics")
            wrapper.log_metrics(metrics)
            wrapper.finish_logging()

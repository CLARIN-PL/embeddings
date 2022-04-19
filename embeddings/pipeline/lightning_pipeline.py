from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar

import srsly

from embeddings.data.datamodule import BaseDataModule, Data
from embeddings.evaluator.evaluator import Evaluator
from embeddings.model.model import Model
from embeddings.pipeline.pipeline import Pipeline
from embeddings.utils.loggers import LightningLoggingConfig, WandbWrapper
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
    ) -> None:
        self.datamodule = datamodule
        self.model = model
        self.evaluator = evaluator
        self.output_path = output_path
        self.logging_config = logging_config

    def run(self, run_name: Optional[str] = None) -> EvaluationResult:
        if run_name:
            run_name = standardize_name(run_name)
        self._save_artifacts()
        model_result = self.model.execute(data=self.datamodule, run_name=run_name)
        result = self.evaluator.evaluate(model_result)
        self._finish_logging()
        return result

    def _save_artifacts(self) -> None:
        srsly.write_json(self.output_path.joinpath("packages.json"), get_installed_packages())

    def _finish_logging(self) -> None:
        if self.logging_config.use_wandb():
            logger = WandbWrapper()
            logger.log_output(
                self.output_path, ignore={"wandb", "csv", "tensorboard", "checkpoints"}
            )
            logger.finish_logging()

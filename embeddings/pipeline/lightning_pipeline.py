from typing import Any, Dict, Generic, Optional, TypeVar

from embeddings.data.datamodule import BaseDataModule, Data
from embeddings.evaluator.evaluator import Evaluator
from embeddings.model.model import Model
from embeddings.pipeline.pipeline import Pipeline

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
    DEFAULT_LOGGING_KWARGS: Dict[str, Any] = {
        "use_tensorboard": True,
        "use_wandb": True,
        "use_csv": True,
    }
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
    ) -> None:
        self.datamodule = datamodule
        self.model = model
        self.evaluator = evaluator

    def run(self, run_name: Optional[str] = None) -> EvaluationResult:
        model_result = self.model.execute(data=self.datamodule, run_name=run_name)
        return self.evaluator.evaluate(model_result)

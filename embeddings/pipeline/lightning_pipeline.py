from typing import Generic, TypeVar

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
    DEFAULT_TASK_TRAIN_KWARGS = {"devices": "auto", "accelerator": "auto"}
    DEFAULT_TASK_MODEL_KWARGS = {"use_scheduler": True}
    DEFAULT_DATAMODULE_KWARGS = {"max_seq_length": None}
    DEFAULT_MODEL_CONFIG_KWARGS = {"classifier_dropout": None}
    DEFAULT_LOGGING_KWARGS = {"use_tensorboard": True, "use_wandb": True, "use_csv": True}

    def __init__(
        self,
        datamodule: BaseDataModule[Data],
        model: Model[BaseDataModule[Data], ModelResult],
        evaluator: Evaluator[ModelResult, EvaluationResult],
    ) -> None:
        self.datamodule = datamodule
        self.model = model
        self.evaluator = evaluator

    def run(self) -> EvaluationResult:
        model_result = self.model.execute(data=self.datamodule)
        return self.evaluator.evaluate(model_result)

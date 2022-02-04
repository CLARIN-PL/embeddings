from typing import Generic, Optional, TypeVar

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
    def __init__(
        self,
        datamodule: BaseDataModule[Data],
        model: Model[BaseDataModule[Data], ModelResult],
        evaluator: Evaluator[ModelResult, EvaluationResult],
    ) -> None:
        self.datamodule = datamodule
        self.model = model
        self.evaluator = evaluator
        self.n_completed_epochs: Optional[int] = None
        self.best_epoch: Optional[int] = None

    def run(self) -> EvaluationResult:
        model_result = self.model.execute(data=self.datamodule)
        self.n_completed_epochs = self.model.task.current_epoch
        self.best_epoch = self.model.best_epoch
        return self.evaluator.evaluate(model_result)

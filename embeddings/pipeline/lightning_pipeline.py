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

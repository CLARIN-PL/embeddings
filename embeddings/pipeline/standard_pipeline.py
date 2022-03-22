from typing import Generic, TypeVar

from embeddings.data.data_loader import DataLoader
from embeddings.data.dataset import BaseDataset
from embeddings.evaluator.evaluator import Evaluator
from embeddings.model.model import Model
from embeddings.pipeline.pipeline import Pipeline
from embeddings.transformation.transformation import Transformation

EvaluationResult = TypeVar("EvaluationResult")
Data = TypeVar("Data")
LoaderResult = TypeVar("LoaderResult")
TransformationResult = TypeVar("TransformationResult")
ModelResult = TypeVar("ModelResult")


class StandardPipeline(
    Pipeline[EvaluationResult],
    Generic[Data, LoaderResult, TransformationResult, ModelResult, EvaluationResult],
):
    def __init__(
        self,
        dataset: BaseDataset[Data],
        data_loader: DataLoader[Data, LoaderResult],
        transformation: Transformation[LoaderResult, TransformationResult],
        model: Model[TransformationResult, ModelResult],
        evaluator: Evaluator[ModelResult, EvaluationResult],
    ) -> None:
        self.dataset = dataset
        self.data_loader = data_loader
        self.transformation = transformation
        self.model = model
        self.evaluator = evaluator

    def run(self) -> EvaluationResult:
        loaded_data = self.data_loader.load(self.dataset)
        transformed_data = self.transformation.transform(loaded_data)
        model_result = self.model.execute(transformed_data)
        return self.evaluator.evaluate(model_result)

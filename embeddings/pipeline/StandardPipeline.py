from typing import TypeVar, Generic
from abc import ABC
from embeddings.pipeline.Pipeline import Pipeline
from embeddings.data.Dataset import Dataset
from embeddings.data.DataLoader import DataLoader
from embeddings.transformation.Transformation import Transformation
from embeddings.model.Model import Model
from embeddings.task.Task import Task
from embeddings.evaluator.Evaluator import Evaluator

EvaluationResult = TypeVar("EvaluationResult")
Data = TypeVar("Data")
LoaderResult = TypeVar("LoaderResult")
TransformationResult = TypeVar("TransformationResult")
ModelResult = TypeVar("ModelResult")
TaskResult = TypeVar("TaskResult")


class StandardPipeline(
    Pipeline[EvaluationResult],
    Generic[Data, LoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult],
):
    def __init__(
        self,
        data_set: Dataset[Data],
        data_loader: DataLoader[Data, LoaderResult],
        transformation: Transformation[LoaderResult, TransformationResult],
        model: Model[TransformationResult, ModelResult],
        task: Task[ModelResult, TaskResult],
        evaluator: Evaluator[TaskResult, EvaluationResult],
    ) -> None:
        self.data_set = data_set
        self.data_loader = data_loader
        self.transformation = transformation
        self.model = model
        self.task = task
        self.evaluator = evaluator

    def run(self) -> EvaluationResult:
        loaded_data = self.data_loader.load(self.data_set)
        transformed_data = self.transformation.transform(loaded_data)
        model_result = self.model.model(transformed_data)
        task_result = self.task.task(model_result)
        return self.evaluator.evaluate(task_result)

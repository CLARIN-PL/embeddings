from typing import TypeVar, Generic
from abc import ABC
from embeddings.pipeline.Pipeline import Pipeline
from embeddings.pipeline.StandardPipeline import StandardPipeline
from embeddings.data.Dataset import Dataset
from embeddings.data.DataLoader import DataLoader
from embeddings.transformation.Transformation import Transformation
from embeddings.model.Model import Model
from embeddings.task.Task import Task
from embeddings.evaluator.Evaluator import Evaluator
from typing import Optional
from copy import deepcopy

Data = TypeVar("Data")
CreationData = TypeVar("CreationData")
LoaderResult = TypeVar("LoaderResult")
CreationLoaderResult = TypeVar("CreationLoaderResult")
TransformationResult = TypeVar("TransformationResult")
CreationTransformationResult = TypeVar("CreationTransformationResult")
ModelResult = TypeVar("ModelResult")
CreationModelResult = TypeVar("CreationModelResult")
TaskResult = TypeVar("TaskResult")
CreationTaskResult = TypeVar("CreationTaskResult")
EvaluationResult = TypeVar("EvaluationResult")
CreationEvaluationResult = TypeVar("CreationEvaluationResult")


class PipelineBuilder(
    ABC,
    Generic[Data, LoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult],
):
    def __init__(
        self,
        dataset: Optional[Dataset[Data]] = None,
        data_loader: Optional[DataLoader[Data, LoaderResult]] = None,
        transformation: Optional[Transformation[LoaderResult, TransformationResult]] = None,
        model: Optional[Model[TransformationResult, ModelResult]] = None,
        task: Optional[Task[ModelResult, TaskResult]] = None,
        evaluator: Optional[Evaluator[TaskResult, EvaluationResult]] = None,
    ) -> None:
        self.dataset = dataset
        self.data_loader = data_loader
        self.transformation = transformation
        self.model = model
        self.task = task
        self.evaluator = evaluator

    @staticmethod
    def with_dataset(
        dataset: Dataset[CreationData],
    ) -> "PipelineBuilder[CreationData, LoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult]":
        return PipelineBuilder(dataset)

    def with_loader(
        self: "PipelineBuilder[Data, LoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult]",
        data_loader: DataLoader[Data, CreationLoaderResult],
    ) -> "PipelineBuilder[Data, CreationLoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult]":
        return PipelineBuilder(
            self.dataset, data_loader, None, self.model, self.task, self.evaluator
        )

    def with_transformation(
        self: "PipelineBuilder[Data, LoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult]",
        transformation: Transformation[LoaderResult, CreationTransformationResult],
    ) -> "PipelineBuilder[Data, LoaderResult, CreationTransformationResult, ModelResult, TaskResult, EvaluationResult]":
        return PipelineBuilder(
            self.dataset, self.data_loader, transformation, None, self.task, self.evaluator
        )

    def with_model(
        self: "PipelineBuilder[Data, LoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult]",
        model: Model[TransformationResult, CreationModelResult],
    ) -> "PipelineBuilder[Data, LoaderResult, TransformationResult, CreationModelResult, TaskResult, EvaluationResult]":
        return PipelineBuilder(
            self.dataset, self.data_loader, self.transformation, model, None, self.evaluator
        )

    def with_task(
        self: "PipelineBuilder[Data, LoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult]",
        task: Task[ModelResult, CreationTaskResult],
    ) -> "PipelineBuilder[Data, LoaderResult, TransformationResult, ModelResult, CreationTaskResult, EvaluationResult]":
        return PipelineBuilder(
            self.dataset, self.data_loader, self.transformation, self.model, task, None
        )

    def with_evaluator(
        self: "PipelineBuilder[Data, LoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult]",
        evaluator: Evaluator[TaskResult, CreationEvaluationResult],
    ) -> "PipelineBuilder[Data, LoaderResult, TransformationResult, ModelResult, TaskResult, CreationEvaluationResult]":
        return PipelineBuilder(
            self.dataset, self.data_loader, self.transformation, self.model, self.task, evaluator
        )

    def build(
        self: "PipelineBuilder[Data, LoaderResult, TransformationResult, ModelResult, TaskResult, EvaluationResult]",
    ) -> Pipeline[EvaluationResult]:
        assert self.dataset is not None
        assert self.data_loader is not None
        assert self.transformation is not None
        assert self.model is not None
        assert self.task is not None
        assert self.evaluator is not None
        return StandardPipeline(
            self.dataset,
            self.data_loader,
            self.transformation,
            self.model,
            self.task,
            self.evaluator,
        )

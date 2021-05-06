from typing import List
from embeddings.pipeline.PipelineBuilder import PipelineBuilder
from embeddings.data.Dataset import Dataset
from embeddings.data.DataLoader import DataLoader, Input, Output
from embeddings.transformation.Transformation import Transformation
from embeddings.model.BaseModel import BaseModel
from embeddings.embedding.Embedding import Embedding
from embeddings.task.Task import Task
from embeddings.evaluator.Evaluator import Evaluator


class DummyDataset(Dataset[str]):
    pass


class DummyLoader(DataLoader[str, str]):
    def load(self, dataset: Dataset[Input]) -> Output:
        pass


class DummyTransformation(Transformation[str, int]):
    def transform(self, data: Input) -> Output:
        pass


class DummyEmbedding(Embedding[int, float]):
    def embed(self, data: Input) -> Output:
        pass


class DummyTask(Task[float, int]):
    def task(self, data: Input) -> Output:
        pass


class DummyEvaluator(Evaluator[int, List[int]]):
    def evaluate(self, data: Input) -> Output:
        pass


def test_pipeline_builder() -> None:
    dataset = DummyDataset()
    data_loader = DummyLoader()
    data_transformation = DummyTransformation()
    task = DummyTask()
    embedding = DummyEmbedding()
    model = BaseModel(embedding, task)
    evaluator = DummyEvaluator()
    pipeline = (
        PipelineBuilder.with_dataset(dataset)
        .with_loader(data_loader)
        .with_transformation(data_transformation)
        .with_model(model)
        .with_evaluator(evaluator)
        .build()
    )
    pipeline.run()

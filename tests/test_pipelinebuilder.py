import os
import tempfile
from typing import Dict

from embeddings.data.data_loader import DataLoader, Input, Output
from embeddings.data.dataset import Dataset
from embeddings.embedding.embedding import Embedding
from embeddings.evaluator.evaluator import Evaluator
from embeddings.model.base_model import BaseModel
from embeddings.pipeline.pipeline_builder import PipelineBuilder
from embeddings.task.task import Task
from embeddings.transformation.transformation import Transformation
from embeddings.utils.json_dict_persister import JsonPersister


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
    def fit_predict(self, data: Input) -> Output:
        pass


class DummyEvaluator(Evaluator[int, Dict[str, int]]):
    def evaluate(self, data: Input) -> Output:
        pass


def test_pipeline_builder() -> None:
    temp_file: str = os.path.join(tempfile.gettempdir(), "pipelinebuilder.embeddings.json")
    dataset = DummyDataset()
    data_loader = DummyLoader()
    data_transformation = DummyTransformation()
    task = DummyTask()
    embedding = DummyEmbedding()
    model = BaseModel(embedding, task)
    evaluator = DummyEvaluator().persisting(JsonPersister(temp_file))
    pipeline = (
        PipelineBuilder.with_dataset(dataset)
        .with_loader(data_loader)
        .with_transformation(data_transformation)
        .with_model(model)
        .with_evaluator(evaluator)
        .build()
    )
    pipeline.run()

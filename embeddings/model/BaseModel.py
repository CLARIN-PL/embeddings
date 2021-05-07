import abc
from typing import TypeVar, Generic
from embeddings.model.Model import Model
from embeddings.task.Task import Task
from embeddings.embedding.Embedding import Embedding

Input = TypeVar("Input")
EmbeddingResult = TypeVar("EmbeddingResult")
Output = TypeVar("Output")


class BaseModel(Model[Input, Output], Generic[Input, EmbeddingResult, Output]):
    def __init__(
        self, embedding: Embedding[Input, EmbeddingResult], task: Task[EmbeddingResult, Output]
    ) -> None:
        self.embedding = embedding
        self.task = task

    def model(self, data: Input) -> Output:
        embedded = self.embedding.embed(data)
        return self.task.task(embedded)

from typing import Generic, TypeVar

from embeddings.embedding.embedding import Embedding
from embeddings.model.model import Model
from embeddings.task.task import Task

Input = TypeVar("Input")
EmbeddingResult = TypeVar("EmbeddingResult")
Output = TypeVar("Output")


class BaseModel(Model[Input, Output], Generic[Input, EmbeddingResult, Output]):
    def __init__(
        self, embedding: Embedding[Input, EmbeddingResult], task: Task[EmbeddingResult, Output]
    ) -> None:
        self.embedding = embedding
        self.task = task

    def execute(self, data: Input) -> Output:
        embedded = self.embedding.embed(data)
        return self.task.fit_predict(embedded)

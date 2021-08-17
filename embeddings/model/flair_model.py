from typing import Dict

import numpy as np
from flair.data import Corpus

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.model.model import Model
from embeddings.task.flair_task.flair_task import FlairTask


class FlairModel(Model[Corpus, Dict[str, np.ndarray]]):
    def __init__(self, embedding: FlairEmbedding, task: FlairTask) -> None:
        super().__init__()
        self.embedding = embedding
        self.task = task

    def execute(self, data: Corpus) -> Dict[str, np.ndarray]:
        self.task.build_task_model(
            embedding=self.embedding, y_dictionary=self.task.make_y_dictionary(data)
        )
        return self.task.fit_predict(data)

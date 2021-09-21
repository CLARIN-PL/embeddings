from typing import Dict, Literal

import numpy as np
from flair.data import Corpus

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.model.model import Model
from embeddings.task.flair_task.flair_task import FlairTask


class FlairModel(Model[Corpus, Dict[str, np.ndarray]]):
    def __init__(
        self,
        embedding: FlairEmbedding,
        task: FlairTask,
        predict_subset: Literal["dev", "test"] = "test",
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.task = task
        self.predict_subset = predict_subset

    def execute(self, data: Corpus) -> Dict[str, np.ndarray]:
        self.task.build_task_model(
            embedding=self.embedding, y_dictionary=self.task.make_y_dictionary(data)
        )
        return self.task.fit_predict(data, self.predict_subset)

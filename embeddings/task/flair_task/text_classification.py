from typing import Any, Dict, List, Optional

import numpy as np
from flair.data import Corpus, Dictionary, Sentence
from flair.models import TextClassifier

from embeddings.data.io import T_path
from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.task.flair_task.flair_task import FlairTask


class TextClassification(FlairTask):
    def __init__(
        self,
        output_path: T_path,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(output_path, task_train_kwargs)
        self.task_model_kwargs = task_model_kwargs if task_model_kwargs else {}

    def build_task_model(self, embedding: FlairEmbedding, y_dictionary: Dictionary) -> None:
        self.model = TextClassifier(
            document_embeddings=embedding.model,
            label_dictionary=y_dictionary,
            **self.task_model_kwargs
        )

    @property
    def y_type(self) -> Any:
        if self.model:
            return self.model.label_type
        else:
            raise self.MODEL_UNDEFINED_EXCEPTION

    @staticmethod
    def make_y_dictionary(data: Corpus, y_type: Optional[str] = None) -> Dictionary:
        return data.make_label_dictionary(y_type)

    @property
    def y_dictionary(self) -> Dictionary:
        if self.model:
            return self.model.label_dictionary
        else:
            raise self.MODEL_UNDEFINED_EXCEPTION

    @staticmethod
    def get_y(data: List[Sentence], y_type: str, y_dictionary: Dictionary) -> np.ndarray:
        labels = [sentence.get_labels(y_type) for sentence in data]
        labels = [[label.value for label in sentence] for sentence in labels]
        labels = [y_dictionary.get_idx_for_item(labels[0]) for labels in labels]
        return np.array(labels)

    @staticmethod
    def remove_labels_from_data(data: List[Sentence], y_type: str) -> None:
        for sentence in data:
            sentence.remove_labels(y_type)

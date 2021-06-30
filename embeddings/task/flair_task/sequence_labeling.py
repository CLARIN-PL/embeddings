from typing import Any, Dict, List, Optional

import numpy as np
from flair.data import Corpus, Dictionary, Sentence
from flair.models import SequenceTagger

from embeddings.data.io import T_path
from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.task.flair_task.flair_task import FlairTask


class SequenceLabeling(FlairTask):
    def __init__(
        self,
        output_path: T_path,
        hidden_size: int,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(output_path, task_train_kwargs)
        self.hidden_size = hidden_size
        self.task_model_kwargs = task_model_kwargs if task_model_kwargs else {}

    def build_task_model(self, embedding: FlairEmbedding, y_dictionary: Dictionary) -> None:
        self.model = SequenceTagger(
            hidden_size=self.hidden_size,
            embeddings=embedding.model,
            tag_dictionary=y_dictionary,
            tag_type="tag",
            **self.task_model_kwargs
        )

    @property
    def y_type(self) -> Any:
        if self.model:
            return self.model.tag_type
        else:
            raise self.MODEL_UNDEFINED_EXCEPTION

    @staticmethod
    def make_y_dictionary(data: Corpus, y_type: Optional[str] = None) -> Dictionary:
        return data.make_tag_dictionary(y_type)

    @property
    def y_dictionary(self) -> Dictionary:
        if self.model:
            return self.model.tag_dictionary
        else:
            raise self.MODEL_UNDEFINED_EXCEPTION

    @staticmethod
    def get_y(data: List[Sentence], y_type: str, y_dictionary: Dictionary) -> np.ndarray:
        y = []
        for sent in data:
            sent_y = []
            for token in sent:
                sent_y.append(token.get_tag(y_type).value)
            y.append(sent_y)
        return np.array(y, dtype=object)

    @staticmethod
    def remove_labels_from_data(data: List[Sentence], y_type: str) -> None:
        for sent in data:
            for token in sent:
                token.remove_labels(y_type)

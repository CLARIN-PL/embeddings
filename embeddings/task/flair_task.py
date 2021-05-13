import abc
from typing import Any, Dict, Optional, List

import numpy as np
from flair.data import Sentence, Corpus
from flair.models import TextClassifier

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.task.task import Task


def remove_flair_predictions_from_data(data: List[Sentence], pred_label: str = "predicted") -> None:
    for sentence in data:
        sentence.remove_labels(pred_label)


def get_flair_labels_from_data(data: List[Sentence], label: str, label_dict: Any) -> np.ndarray:
    labels = [sentence.get_labels(label) for sentence in data]
    labels = [[label.value for label in sentence] for sentence in labels]
    labels = [label_dict.get_idx_for_item(labels[0]) for labels in labels]
    return np.array(labels)


class FlairTask(Task[Corpus, Dict[str, np.ndarray]]):
    @abc.abstractmethod
    def build_task_model(self, embedding: FlairEmbedding, label_dictionary: Any) -> None:
        pass


class FlairTextClassification(FlairTask):
    def __init__(
        self,
        task_model_args: Optional[Dict[str, Any]] = None,
        task_trainer_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.task_model_args = task_model_args if task_model_args else {}
        self.task_trainer_args = task_trainer_args if task_trainer_args else {}
        self.model = None

    def build_task_model(self, embedding: FlairEmbedding, label_dictionary: Any) -> None:
        self.model = TextClassifier(
            document_embeddings=embedding.model,
            label_dictionary=label_dictionary,
            **self.task_model_args
        )

    def task(self, data: Corpus) -> Dict[str, np.ndarray]:
        if not self.model:
            raise ValueError("Model undefined. Use build_task_model() first!")

        self.model.fit(data.train, **self.task_trainer_args)
        y_pred = self.predict(data.test)
        y_true = get_flair_labels_from_data(
            data.test, label=self.model.label_type, label_dict=self.model.label_dict
        )
        return {"y_true": y_true, "y_pred": y_pred}

    def predict(self, data: List[Sentence], mini_batch_size: int = 32) -> np.ndarray:
        if not self.model:
            raise ValueError("Model undefined. Use build_task_model() first!")

        remove_flair_predictions_from_data(data)

        self.model.predict(
            sentences=data,
            mini_batch_size=mini_batch_size,
            label_name="predicted",
            return_loss=True,
        )

        y_pred = get_flair_labels_from_data(
            data, label="predicted", label_dict=self.model.label_dict
        )
        remove_flair_predictions_from_data(data)
        return y_pred

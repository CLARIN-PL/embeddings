import abc
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import flair
import numpy as np
from flair.data import Sentence, Corpus
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.task.task import Task
from experimental.data.io import T_path
from experimental.defaults import RESULTS_PATH


def remove_flair_predictions_from_data(data: List[Sentence], pred_label: str = "predicted") -> None:
    for sentence in data:
        sentence.remove_labels(pred_label)


def get_flair_labels_from_data(
    data: List[Sentence], label: str, label_dictionary: Any
) -> np.ndarray:
    labels = [sentence.get_labels(label) for sentence in data]
    labels = [[label.value for label in sentence] for sentence in labels]
    labels = [label_dictionary.get_idx_for_item(labels[0]) for labels in labels]
    return np.array(labels)


class FlairTask(Task[Corpus, Dict[str, np.ndarray]]):
    def __init__(
        self,
        output_path: T_path = RESULTS_PATH,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model: Optional[flair.nn.Model] = None
        self.output_path: Path = Path(output_path)
        self.task_train_kwargs = task_train_kwargs if task_train_kwargs else {}

    def fit(
        self,
        corpus: Corpus,
    ) -> Dict[Any, Any]:
        self.trainer = ModelTrainer(self.model, corpus)
        log: Dict[Any, Any] = self.trainer.train(
            base_path=self.output_path, **self.task_train_kwargs
        )
        return log

    @abc.abstractmethod
    def build_task_model(self, embedding: FlairEmbedding, label_dictionary: Any) -> None:
        pass


class FlairTextClassification(FlairTask):
    def __init__(
        self,
        output_path: T_path,
        task_model_kwargs: Optional[Dict[str, Any]] = None,
        task_train_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(output_path, task_train_kwargs)
        self.task_model_kwargs = task_model_kwargs if task_model_kwargs else {}

    def build_task_model(self, embedding: FlairEmbedding, label_dictionary: Any) -> None:
        self.model = TextClassifier(
            document_embeddings=embedding.model,
            label_dictionary=label_dictionary,
            **self.task_model_kwargs
        )

    def task(self, data: Corpus) -> Dict[str, np.ndarray]:
        if not self.model:
            raise ValueError("Model undefined. Use build_task_model() first!")

        self.fit(data)
        y_pred, _ = self.predict(data.test)
        y_true = get_flair_labels_from_data(
            data.test, label=self.model.label_type, label_dictionary=self.model.label_dictionary
        )
        return {"y_true": y_true, "y_pred": y_pred}

    def predict(self, data: List[Sentence], mini_batch_size: int = 32) -> Tuple[np.ndarray, float]:
        if not self.model:
            raise ValueError("Model undefined. Use build_task_model() first!")

        remove_flair_predictions_from_data(data)

        loss = self.model.predict(
            sentences=data,
            mini_batch_size=mini_batch_size,
            label_name="predicted",
            return_loss=True,
        )

        y_pred = get_flair_labels_from_data(
            data, label="predicted", label_dictionary=self.model.label_dictionary
        )
        remove_flair_predictions_from_data(data)
        return y_pred, loss

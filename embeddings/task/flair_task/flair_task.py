import abc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flair
import numpy as np
from flair.data import Corpus, Dictionary, Sentence
from flair.trainers import ModelTrainer

from embeddings.data.io import T_path
from embeddings.defaults import RESULTS_PATH
from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.task.task import Task


class FlairTask(Task[Corpus, Dict[str, np.ndarray]]):
    MODEL_UNDEFINED_EXCEPTION = ValueError("Model undefined. Use build_task_model() first!")

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

    def predict(self, data: List[Sentence], mini_batch_size: int = 32) -> Tuple[np.ndarray, float]:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION

        self.remove_labels_from_data(data, "predicted")

        loss = self.model.predict(
            sentences=data,
            mini_batch_size=mini_batch_size,
            label_name="predicted",
            return_loss=True,
        )

        y_pred = self.get_y(data, y_type="predicted", y_dictionary=self.y_dictionary)
        self.remove_labels_from_data(data, "predicted")
        return y_pred, loss

    def fit_predict(self, data: Corpus) -> Dict[str, np.ndarray]:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION

        self.fit(data)
        y_pred, _ = self.predict(data.test)
        y_true = self.get_y(data.test, self.y_type, self.y_dictionary)
        return {"y_true": y_true, "y_pred": y_pred}

    @property
    @abc.abstractmethod
    def y_type(self) -> Any:
        pass

    @abc.abstractmethod
    def build_task_model(self, embedding: FlairEmbedding, y_dictionary: Dictionary) -> None:
        pass

    @staticmethod
    @abc.abstractmethod
    def make_y_dictionary(data: Corpus, y_type: Optional[str] = None) -> Dictionary:
        pass

    @property
    @abc.abstractmethod
    def y_dictionary(self) -> Dictionary:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_y(data: List[Sentence], y_type: str, y_dictionary: Dictionary) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def remove_labels_from_data(data: List[Sentence], y_type: str) -> None:
        pass

import abc
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import flair
import numpy as np
from flair.data import Sentence, Corpus, Dictionary
from flair.models import TextClassifier, SequenceTagger
from flair.trainers import ModelTrainer

from embeddings.embedding.flair_embedding import FlairEmbedding
from embeddings.task.task import Task
from experimental.data.io import T_path
from experimental.defaults import RESULTS_PATH


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

    def task(self, data: Corpus) -> Dict[str, np.ndarray]:
        if not self.model:
            raise self.MODEL_UNDEFINED_EXCEPTION

        self.fit(data)
        y_pred, _ = self.predict(data.test)
        y_true = self.get_y(data.test, self.y_type, self.y_dictionary)
        return {"y_true": y_true, "y_pred": y_pred}

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
    def get_y(data: List[Sentence], y_type: str, y_dictionary: Any) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def remove_labels_from_data(data: List[Sentence], y_type: str) -> None:
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

    def build_task_model(self, embedding: FlairEmbedding, y_dictionary: Any) -> None:
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
    def get_y(data: List[Sentence], y_type: str, y_dictionary: Any) -> np.ndarray:
        labels = [sentence.get_labels(y_type) for sentence in data]
        labels = [[label.value for label in sentence] for sentence in labels]
        labels = [y_dictionary.get_idx_for_item(labels[0]) for labels in labels]
        return np.array(labels)

    @staticmethod
    def remove_labels_from_data(data: List[Sentence], y_type: str) -> None:
        for sentence in data:
            sentence.remove_labels(y_type)


class FlairSequenceTagging(FlairTask):
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

    def build_task_model(self, embedding: FlairEmbedding, y_dictionary: Any) -> None:
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
    def get_y(data: List[Sentence], y_type: str, y_dictionary: Any) -> np.ndarray:
        y = []
        for sent in data:
            sent_y = []
            for token in sent:
                sent_y.append(token.get_tag(y_type).value)
            y.append(sent_y)
        return np.array(y)

    @staticmethod
    def remove_labels_from_data(data: List[Sentence], y_type: str) -> None:
        for sent in data:
            for token in sent:
                token.remove_labels(y_type)

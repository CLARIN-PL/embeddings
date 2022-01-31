from typing import Any, Dict, Optional, Union

from numpy import typing as nptyping
import pandas as pd
from sklearn.base import ClassifierMixin as AnySklearnClassifier

from embeddings.task.sklearn_task.sklearn_task import SklearnTask
from embeddings.embedding.sklearn_embedding import SklearnEmbedding


class TextClassification(SklearnTask):
    def __init__(
        self,
        classifier: AnySklearnClassifier,
        train_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.train_model_kwargs = train_model_kwargs if train_model_kwargs else {}
        self.classifier = classifier(**self.train_model_kwargs)

    def build_task_model(self, embedding: SklearnEmbedding) -> None:
        self.embedding = embedding

    def fit(
        self,
        x_train: Union[pd.DataFrame, nptyping.NDArray[Any]],
        y_train: Union[pd.Series, nptyping.NDArray[Any]],
    ) -> None:

        self.classifier.fit(x_train, y_train)

    def predict(self, x: Union[pd.DataFrame, nptyping.NDArray[Any]]) -> nptyping.NDArray[Any]:
        return self.classifier.predict(x)

    def fit_predict(self, data: Dict[str, Any], predict_subset: str = "test"):
        x_train = self.embedding.embed(data["train"]["x"])
        y_train = data["train"]["y"]

        self.fit(x_train, y_train)

        predictions = self.predict(self.embedding.embed(data[predict_subset]["x"]))

        return predictions

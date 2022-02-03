from typing import Any, Dict, Optional, Union

import pandas as pd
from numpy import typing as nptyping
from sklearn.base import ClassifierMixin as AnySklearnClassifier

from embeddings.embedding.sklearn_embedding import SklearnEmbedding
from embeddings.task.sklearn_task.sklearn_task import SklearnTask


class TextClassification(SklearnTask):
    def __init__(
        self,
        classifier: AnySklearnClassifier,
        classifier_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.classifier_kwargs = classifier_kwargs if classifier_kwargs else {}
        self.classifier = classifier(**self.classifier_kwargs)

    def build_task_model(self, embedding: SklearnEmbedding) -> None:
        self.embedding = embedding

    def fit(
        self,
        x_train: Union[pd.DataFrame, nptyping.NDArray[Any]],
        y_train: Union[pd.Series, nptyping.NDArray[Any]],
    ) -> None:

        self.classifier.fit(x_train, y_train)

    def predict(
        self,
        data: Dict[str, Union[pd.DataFrame, nptyping.NDArray[Any]]],
        predict_subset: str = "test",
    ) -> Dict[str, nptyping.NDArray[Any]]:
        predictions = self.classifier.predict(self.embedding.embed(data[predict_subset]["x"]))
        model_result = {
            "y_pred": predictions,
            "y_true": data[predict_subset]["y"].values,
        }
        return model_result

    def fit_predict(
        self,
        data: Dict[str, Union[pd.DataFrame, nptyping.NDArray[Any]]],
        predict_subset: str = "test",
    ) -> Dict[str, nptyping.NDArray[Any]]:
        x_train = self.embedding.embed(data["train"]["x"])
        y_train = data["train"]["y"]
        self.fit(x_train, y_train)
        return self.predict(data)

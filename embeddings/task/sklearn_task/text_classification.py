from typing import Any, Dict, Optional, Union

from numpy import typing as nptyping
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from typing_extensions import Literal

from embeddings.task.sklearn_task.sklearn_task import SklearnTask


class TextClassification(SklearnTask):
    def __init__(
            self,
            model: Literal["tree", "forest", "logistic"] = "tree",
            train_model_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.classifier = None
        self.train_model_kwargs = train_model_kwargs
        self.model = model

    def fit(
            self,
            x_train: Union[pd.DataFrame, nptyping.NDArray[Any]],
            y_train: Union[pd.Series, nptyping.NDArray[Any]]
    ) -> None:
        fit_kwargs = self.train_model_kwargs if self.train_model_kwargs else {}
        if self.model == "tree":
            self.classifier = DecisionTreeClassifier(**fit_kwargs)
        elif self.model == "forest":
            self.classifier = RandomForestClassifier(**fit_kwargs)
        elif self.model == "logistic":
            self.classifier = LogisticRegression(**fit_kwargs)

        self.classifier.fit(x_train, y_train)

    def predict(
            self,
            x: Union[pd.DataFrame, nptyping.NDArray[Any]]
    ) -> nptyping.NDArray[Any]:
        return self.classifier.predict(x)

    def fit_predict(self, data: Dict[str, Any]):
        x_train = data["train"]["x"]
        y_train = data["train"]["y"]

        self.fit(x_train, y_train)

        predictions = {}
        for subset in data.keys():
            predictions[subset] = self.predict(data[subset]["x"])

        return predictions

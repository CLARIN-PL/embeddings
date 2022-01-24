from typing import Any, Dict, Union

from numpy import typing as nptyping
import pandas as pd

from embeddings.task.task import Task


class SklearnTask(Task[pd.DataFrame, Dict]):
    def __init__(
            self
    ):
        super().__init__()

    def fit(
            self,
            x_train: Union[pd.DataFrame, nptyping.NDArray[Any]],
            y_train: Union[pd.Series, nptyping.NDArray[Any]]
    ):
        pass

    def predict(
            self,
            x: Union[pd.DataFrame, nptyping.NDArray[Any]]
    ):
        pass

    def fit_predict(self, data):
        super().fit_predict(data)

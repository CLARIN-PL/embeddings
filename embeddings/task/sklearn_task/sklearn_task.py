from typing import Any, Dict, Union

import pandas as pd
from numpy import typing as nptyping

from embeddings.task.task import Task


class SklearnTask(Task[pd.DataFrame, Dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        x_train: Union[pd.DataFrame, nptyping.NDArray[Any]],
        y_train: Union[pd.Series, nptyping.NDArray[Any]],
    ) -> None:
        pass

    def predict(self, x: Union[pd.DataFrame, nptyping.NDArray[Any]]) -> None:
        pass

    def fit_predict(self, data: Any) -> Dict[str, Any]:
        super().fit_predict(data)

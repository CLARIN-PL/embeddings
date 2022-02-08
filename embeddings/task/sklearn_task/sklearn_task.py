from abc import abstractmethod
from typing import Any, Dict

import pandas as pd
from numpy import typing as nptyping

from embeddings.task.task import Task
from embeddings.utils.array_like import ArrayLike


class SklearnTask(Task[pd.DataFrame, Dict[str, Any]]):
    @abstractmethod
    def fit(
        self,
        data: Dict[str, ArrayLike],
    ) -> None:
        pass

    @abstractmethod
    def predict(
        self,
        data: Dict[str, ArrayLike],
        predict_subset: str = "test",
    ) -> Dict[str, nptyping.NDArray[Any]]:
        pass

    @abstractmethod
    def fit_predict(
        self, data: Dict[str, ArrayLike], predict_subset: str = "test"
    ) -> Dict[str, nptyping.NDArray[Any]]:
        pass

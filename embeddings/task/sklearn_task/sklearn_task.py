from abc import abstractmethod
from typing import Dict

import pandas as pd

from embeddings.evaluator.evaluation_results import Predictions
from embeddings.task.task import Task
from embeddings.utils.array_like import ArrayLike


class SklearnTask(Task[pd.DataFrame, Predictions]):
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
    ) -> Predictions:
        pass

    @abstractmethod
    def fit_predict(self, data: Dict[str, ArrayLike], predict_subset: str = "test") -> Predictions:
        pass

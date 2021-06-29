import abc
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch

from embeddings.evaluator.evaluator import Evaluator
from embeddings.metric.metric import Metric


class MetricsEvaluator(Evaluator[Dict[str, np.ndarray], Dict[str, Any]]):
    @property
    @abc.abstractmethod
    def metrics(
        self,
    ) -> Sequence[Metric[Union[List[Any], np.ndarray, torch.Tensor], Dict[Any, Any]]]:
        pass

    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {
            str(metric): metric.compute(y_true=data["y_true"], y_pred=data["y_pred"])
            for metric in self.metrics
        }

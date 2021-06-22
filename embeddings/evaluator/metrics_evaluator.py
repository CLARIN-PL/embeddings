import abc
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import Metric

from embeddings.evaluator.evaluator import Evaluator


class MetricsEvaluator(Evaluator[Dict[str, np.ndarray], Dict[str, Any]]):
    @property
    @abc.abstractmethod
    def metrics(self) -> List[Tuple[Metric, Dict[str, Any]]]:
        pass

    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {
            metric.name: metric.compute(
                references=data["y_true"], predictions=data["y_pred"], **kwargs
            )
            for metric, kwargs in self.metrics
        }

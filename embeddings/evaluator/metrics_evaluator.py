import abc
from typing import List, Tuple, Dict, Any

import numpy as np
from datasets import Metric

from embeddings.evaluator.evaluator import Evaluator


class MetricsEvaluator(Evaluator[Dict[str, np.ndarray], List[Any]]):
    @property
    @abc.abstractmethod
    def metrics(self) -> List[Tuple[Metric, Dict[str, Any]]]:
        pass

    def evaluate(self, data: Dict[str, np.ndarray]) -> List[Any]:
        return [
            metric.compute(references=data["y_true"], predictions=data["y_pred"], **kwargs)
            for metric, kwargs in self.metrics
        ]
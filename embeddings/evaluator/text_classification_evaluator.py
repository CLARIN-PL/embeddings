from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from datasets import load_metric, Metric

from embeddings.evaluator.evaluator import Evaluator


class TextClassificationEvaluator(Evaluator[Dict[str, np.ndarray], List[Any]]):
    def __init__(self, metrics: Optional[List[Tuple[Metric, Dict[str, Any]]]] = None):
        super().__init__()
        self.metrics = metrics if metrics else self._get_default_metrics()

    @staticmethod
    def _get_default_metrics() -> List[Tuple[Metric, Dict[str, Any]]]:
        return [
            (load_metric("accuracy"), {}),
            (load_metric("f1"), {"average": "macro"}),
            (load_metric("recall"), {"average": "macro"}),
            (load_metric("precision"), {"average": "macro"}),
        ]

    def evaluate(self, data: Dict[str, np.ndarray]) -> List[Any]:
        return [
            metric.compute(references=data["y_true"], predictions=data["y_pred"], **kwargs)
            for metric, kwargs in self.metrics
        ]

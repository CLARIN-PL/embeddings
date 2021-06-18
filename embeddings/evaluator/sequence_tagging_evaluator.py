from typing import List, Tuple, Dict, Any, Union

import datasets
import numpy as np

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metrics import metrics
from embeddings.metrics.pos import POSTaggingMetric


class SequenceTaggingEvaluator(MetricsEvaluator):
    @property
    def metrics(self) -> List[Tuple[datasets.Metric, Dict[str, Any]]]:
        return [
            (datasets.load_metric("seqeval"), {}),
        ]


class POSTaggingEvaluator(MetricsEvaluator):
    @property
    def metrics(
        self,
    ) -> List[
        Tuple[Union[datasets.Metric, metrics.Metric[np.ndarray, Dict[str, Any]]], Dict[str, Any]]
    ]:
        return [(POSTaggingMetric(), {})]

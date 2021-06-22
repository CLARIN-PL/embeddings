from typing import Any, Dict, List, Tuple

import datasets
import numpy as np

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric import metric
from embeddings.metric.pos_metric import POSTaggingSeqevalMetric


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
    ) -> List[Tuple[metric.Metric[np.ndarray, Dict[str, Any]], Dict[str, Any]]]:
        return [(POSTaggingSeqevalMetric(), {})]

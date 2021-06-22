from typing import Any, Dict, List, Tuple

import datasets

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
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
    ) -> List[Tuple[POSTaggingSeqevalMetric, Dict[str, Any]]]:
        return [(POSTaggingSeqevalMetric(), {})]

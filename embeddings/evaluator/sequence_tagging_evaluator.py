from typing import Any, Dict, List, Tuple

from datasets import Metric, load_metric

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator


class SequenceTaggingEvaluator(MetricsEvaluator):
    @property
    def metrics(self) -> List[Tuple[Metric, Dict[str, Any]]]:
        return [
            (load_metric("seqeval"), {}),
        ]

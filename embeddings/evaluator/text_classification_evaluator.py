from typing import List, Tuple, Dict, Any

from datasets import Metric, load_metric

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator


class TextClassificationEvaluator(MetricsEvaluator):
    @property
    def metrics(self) -> List[Tuple[Metric, Dict[str, Any]]]:
        return [
            (load_metric("accuracy"), {}),
            (load_metric("f1"), {"average": "macro"}),
            (load_metric("recall"), {"average": "macro"}),
            (load_metric("precision"), {"average": "macro"}),
        ]

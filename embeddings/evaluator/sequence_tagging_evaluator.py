from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
from datasets import Metric, load_metric

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from itertools import chain


class SequenceTaggingEvaluator(MetricsEvaluator):
    def __init__(self, seqeval_args: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.seqeval_args = seqeval_args if seqeval_args else self._get_seqeval_kwargs()

    def _get_seqeval_kwargs(self) -> Dict[str, Any]:
        return {"suffix": None, "scheme": "BILOU", "mode": "strict"}

    @property
    def metrics(self) -> List[Tuple[Metric, Dict[str, Any]]]:
        return [
            (load_metric("seqeval"), self.seqeval_args),
        ]

    @staticmethod
    def _convert_single_tag_to_bilou_scheme(
        input_arr: List[Union[List[str], np.ndarray]]
    ) -> List[List[str]]:
        return [[f"U-{tag}" for tag in tags] for tags in input_arr]

    def preprocess_tags(
        self, input_arr: List[Union[List[str], np.ndarray]]
    ) -> List[Union[List[str], np.ndarray]]:
        tags = np.unique(list(chain.from_iterable(input_arr))).tolist()
        if not all([tag.startswith(("B-", "I-", "L-", "O", "U-", "E-", "S-")) for tag in tags]):
            return self._convert_single_tag_to_bilou_scheme(input_arr)

        return input_arr

    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {
            metric.name: metric.compute(
                references=data["y_true"]
                if metric.name != "seqeval"
                else self.preprocess_tags(data["y_true"]),
                predictions=data["y_pred"]
                if metric.name != "seqeval"
                else self.preprocess_tags(data["y_pred"]),
                **kwargs,
            )
            for metric, kwargs in self.metrics
        }

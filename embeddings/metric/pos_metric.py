from itertools import chain
from typing import Any, Dict, List

import numpy as np
from datasets import load_metric

from embeddings.metric import metric


class _POSTaggingSeqevalMetric(metric.Metric[np.ndarray, Dict[str, Any]]):
    ALLOWED_TAG_PREFIXES = {"U-"}

    def __init__(self) -> None:
        name = type(self).__name__
        super().__init__(name)

    @property
    def _seqeval_args(self) -> Dict[str, Any]:
        return {"suffix": None, "scheme": "BILOU", "mode": "strict"}

    @staticmethod
    def _convert_single_tag_to_bilou_scheme(tags: np.ndarray) -> List[List[str]]:
        return [[f"U-{tag}" for tag in sequence] for sequence in tags]

    @staticmethod
    def _have_tags_unit_prefix(tags: np.ndarray) -> bool:
        unique_tags = set(chain.from_iterable(tags))
        tag_prefixes = set(it[0:2] for it in unique_tags)
        return len(tag_prefixes.difference(_POSTaggingSeqevalMetric.ALLOWED_TAG_PREFIXES)) < 1

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        seqeval = load_metric("seqeval")

        if not self._have_tags_unit_prefix(y_pred):
            y_pred = self._convert_single_tag_to_bilou_scheme(y_pred)
        if not self._have_tags_unit_prefix(y_true):
            y_true = self._convert_single_tag_to_bilou_scheme(y_true)

        outs = seqeval.compute(references=y_true, predictions=y_pred, **self._seqeval_args)
        assert isinstance(outs, Dict)
        return outs


class POSTaggingSeqevalMetric:  # ktagowski: Temporal Wrapper to handle y_true, y_pred params
    def __init__(self) -> None:
        self.metric: _POSTaggingSeqevalMetric = _POSTaggingSeqevalMetric()

    @property
    def name(self) -> str:
        return type(self).__name__

    def compute(
        self, predictions: np.ndarray, references: np.ndarray, **kwargs: Any
    ) -> Dict[str, Any]:
        return self.metric.compute(y_true=references, y_pred=predictions, **kwargs)

from itertools import chain
from typing import List, Dict, Any, Union

import numpy as np
from datasets import load_metric

from embeddings.metrics import metrics


class POSTaggingMetric(metrics.Metric[np.ndarray, Dict[str, Any]]):
    ALLOWED_TAG_PREFIXES = {"U-"}

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
        return len(tag_prefixes.difference(POSTaggingMetric.ALLOWED_TAG_PREFIXES)) < 1

    def compute(self, predictions: np.ndarray, references: np.ndarray) -> Dict[str, Any]:
        seqeval = load_metric("seqeval")

        if not self._have_tags_unit_prefix(references):
            references = self._convert_single_tag_to_bilou_scheme(references)
        if not self._have_tags_unit_prefix(predictions):
            predictions = self._convert_single_tag_to_bilou_scheme(predictions)

        outs = seqeval.compute(predictions=predictions, references=references, **self._seqeval_args)
        assert isinstance(outs, dict)
        return outs

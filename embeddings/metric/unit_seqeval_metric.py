from itertools import chain
from typing import Any, Dict, List

import numpy as np

from embeddings.metric.hugging_face_metric import HuggingFaceMetric


class UnitSeqevalMetric(HuggingFaceMetric):
    ALLOWED_TAG_PREFIXES = {"U-"}

    def __init__(self) -> None:
        super().__init__(
            "seqeval", compute_kwargs={"suffix": None, "scheme": "BILOU", "mode": "strict"}
        )

    @staticmethod
    def _convert_single_tag_to_bilou_scheme(tags: np.ndarray) -> List[List[str]]:
        return [[f"U-{tag}" for tag in sequence] for sequence in tags]

    @staticmethod
    def _have_tags_unit_prefix(tags: np.ndarray) -> bool:
        unique_tags = set(chain.from_iterable(tags))
        tag_prefixes = set(it[0:2] for it in unique_tags)
        return not tag_prefixes.difference(UnitSeqevalMetric.ALLOWED_TAG_PREFIXES)

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        if not self._have_tags_unit_prefix(y_pred):
            y_pred = self._convert_single_tag_to_bilou_scheme(y_pred)
        if not self._have_tags_unit_prefix(y_true):
            y_true = self._convert_single_tag_to_bilou_scheme(y_true)

        return super().compute(y_true=y_true, y_pred=y_pred, **kwargs)

    def __str__(self) -> str:
        return "UnitSeqeval"

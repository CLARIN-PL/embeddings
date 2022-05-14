from itertools import chain
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from embeddings.metric.hugging_face_metric import HF_metric_input, HuggingFaceMetric
from embeddings.metric.seqeval_metric import SeqevalMetric


class UnitSeqevalMetric(HuggingFaceMetric):
    ALLOWED_TAG_PREFIXES = {"U-"}

    def __init__(self) -> None:
        super().__init__(
            SeqevalMetric(), compute_kwargs={"suffix": None, "scheme": "BILOU", "mode": "strict"}
        )

    @staticmethod
    def _convert_single_tag_to_bilou_scheme(tags: Optional[HF_metric_input]) -> List[List[str]]:
        assert isinstance(tags, (np.ndarray, list, torch.Tensor))
        return [[f"U-{tag}" if tag != "O" else tag for tag in sequence] for sequence in tags]

    @staticmethod
    def _have_tags_unit_prefix(tags: Optional[HF_metric_input]) -> bool:
        assert isinstance(tags, (np.ndarray, list, torch.Tensor))
        unique_tags = set(chain.from_iterable(tags))
        tag_prefixes = set(it[0:2] for it in unique_tags)
        return not tag_prefixes.difference(UnitSeqevalMetric.ALLOWED_TAG_PREFIXES)

    def compute(
        self, y_true: Optional[HF_metric_input], y_pred: Optional[HF_metric_input], **kwargs: Any
    ) -> Dict[str, Any]:
        if not self._have_tags_unit_prefix(y_pred):
            y_pred = self._convert_single_tag_to_bilou_scheme(y_pred)
        if not self._have_tags_unit_prefix(y_true):
            y_true = self._convert_single_tag_to_bilou_scheme(y_true)

        return super().compute(y_true=y_true, y_pred=y_pred, **kwargs)

    def __str__(self) -> str:
        return "UnitSeqeval"

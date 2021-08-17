from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np
import torch

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.metric import Metric
from embeddings.metric.unit_seqeval_metric import UnitSeqevalMetric


class SequenceLabelingEvaluator(MetricsEvaluator):
    SEQEVAL_EVALUATION_MODES: Set[str] = {"conll", "strict"}

    def __init__(
        self, evaluation_mode: str = "conll", tagging_scheme: Optional[str] = None
    ) -> None:
        super().__init__()
        self.metric = self._get_metric(evaluation_mode, tagging_scheme)

    def _get_metric(
        self, evaluation_mode: str, tagging_scheme: Optional[str] = None
    ) -> Union[HuggingFaceMetric, UnitSeqevalMetric]:
        if evaluation_mode in SequenceLabelingEvaluator.SEQEVAL_EVALUATION_MODES:
            if evaluation_mode == "strict" and not tagging_scheme:
                raise ValueError("Tagging scheme must be set, when using strict evaluation mode!")
            return HuggingFaceMetric(
                name="seqeval",
                compute_kwargs={
                    "mode": evaluation_mode if evaluation_mode == "strict" else None,
                    "scheme": tagging_scheme,
                },
            )
        elif evaluation_mode == "unit":
            return UnitSeqevalMetric()
        else:
            raise ValueError(
                f"Evaluation mode {evaluation_mode} not supported. Must be one of [unit, conll, strict]."
            )

    @property
    def metrics(
        self,
    ) -> Sequence[Metric[Union[List[Any], np.ndarray, torch.Tensor], Dict[Any, Any]]]:
        return [self.metric]

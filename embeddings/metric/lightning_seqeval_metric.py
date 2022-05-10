from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import ClassLabel
from torchmetrics import Metric

from embeddings.evaluator.sequence_labeling_evaluator import (
    EvaluationMode,
    SequenceLabelingEvaluator,
    TaggingScheme,
)
from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.seqeval_metric import SeqevalMetric
from embeddings.metric.unit_seqeval_metric import UnitSeqevalMetric


class SeqEvalMetric(Metric):
    def __init__(
        self,
        class_label: ClassLabel,
        metric_name: str = "f1_macro",
        ignore_index: int = -100,
        evaluation_mode: EvaluationMode = EvaluationMode.CONLL,
        tagging_scheme: Optional[TaggingScheme] = None,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.class_label = class_label
        self.metric_name = metric_name
        self.ignore_index = ignore_index
        self.evaluation_mode = evaluation_mode
        self.tagging_scheme = tagging_scheme
        self.metric = self._get_metric()
        self.add_state("y_pred", default=list(), dist_reduce_fx="cat")
        self.add_state("y_true", default=list(), dist_reduce_fx="cat")

    def _get_metric(
        self,
    ) -> HuggingFaceMetric:
        if self.evaluation_mode in SequenceLabelingEvaluator.SEQEVAL_EVALUATION_MODES:
            if self.evaluation_mode == "strict" and not self.tagging_scheme:
                raise ValueError("Tagging scheme must be set, when using strict evaluation mode!")
            elif self.evaluation_mode == "conll" and self.tagging_scheme:
                raise ValueError("Tagging scheme can be set only in strict mode!")
            return HuggingFaceMetric(
                metric=SeqevalMetric(),
                compute_kwargs={
                    "mode": self.evaluation_mode if self.evaluation_mode == "strict" else None,
                    "scheme": self.tagging_scheme,
                },
            )
        elif self.evaluation_mode == "unit":
            return UnitSeqevalMetric()
        else:
            raise ValueError(
                f"Evaluation mode {self.evaluation_mode} not supported. Must be one of "
                f"[unit, conll, strict]."
            )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        preds, target = self._input_format(preds, target)
        self.y_pred += preds
        self.y_true += target

    def compute(self) -> float:
        result = self.metric.compute(y_true=self.y_true, y_pred=self.y_pred)
        assert isinstance(result, dict)
        metric_value = result[self.metric_name]
        assert isinstance(metric_value, float)
        return metric_value

    def _input_format(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[List[str], List[str]]:
        return (
            [self.class_label.int2str(x.item()) for x in preds[target != self.ignore_index]],
            [self.class_label.int2str(x.item()) for x in target[target != self.ignore_index]],
        )

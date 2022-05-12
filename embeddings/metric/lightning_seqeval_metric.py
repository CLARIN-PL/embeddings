from typing import Dict, List, Optional

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


class SeqevalTorchMetric(Metric):
    def __init__(
        self,
        class_label: ClassLabel,
        average: str = "macro",
        evaluation_mode: EvaluationMode = EvaluationMode.CONLL,
        tagging_scheme: Optional[TaggingScheme] = None,
        dist_sync_on_step: bool = False,
    ) -> None:
        assert average in ["macro", "micro", "weighted"]
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.class_label = class_label
        self.average = average
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

    # ignoring due to different types defined in parent abstract method
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        assert preds.shape == target.shape
        self.y_pred += [preds]
        self.y_true += [target]

    def compute(self) -> Dict[str, float]:
        result = self.metric.compute(**self._input_format(preds=self.y_pred, targets=self.y_true))
        result.pop("classes")
        result = {k: v for k, v in result.items() if self.average in k}
        return result

    def _input_format(
        self, preds: List[torch.Tensor], targets: List[torch.Tensor]
    ) -> Dict[str, List[List[str]]]:
        return {
            "y_true": [self.class_label.int2str(x) for x in preds],
            "y_pred": [self.class_label.int2str(x) for x in targets],
        }

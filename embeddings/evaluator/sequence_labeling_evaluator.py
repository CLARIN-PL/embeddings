from enum import Enum
from typing import Any, Dict, Final, List, Optional, Sequence, Set, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.metric import Metric
from embeddings.metric.unit_seqeval_metric import UnitSeqevalMetric


class SequenceLabelingEvaluator(MetricsEvaluator):
    class EvaluationMode(str, Enum):
        UNIT = "unit"
        CONLL = "conll"
        STRICT = "strict"

    class TaggingScheme(str, Enum):
        IOB1 = "IOB1"
        IOB2 = "IOB2"
        IOE1 = "IOE1"
        IOE2 = "IOE2"
        IOBES = "IOBES"
        BILOU = "BILOU"

    SEQEVAL_EVALUATION_MODES: Final[Set[str]] = {EvaluationMode.CONLL, EvaluationMode.STRICT}

    def __init__(
        self,
        evaluation_mode: EvaluationMode = EvaluationMode.CONLL,
        tagging_scheme: Optional[TaggingScheme] = None,
    ) -> None:
        super().__init__()
        self.metric = self._get_metric(evaluation_mode, tagging_scheme)

    def _get_metric(
        self,
        evaluation_mode: EvaluationMode,
        tagging_scheme: Optional[TaggingScheme] = None,
    ) -> Union[HuggingFaceMetric, UnitSeqevalMetric]:
        if evaluation_mode in SequenceLabelingEvaluator.SEQEVAL_EVALUATION_MODES:
            if evaluation_mode == "strict" and not tagging_scheme:
                raise ValueError("Tagging scheme must be set, when using strict evaluation mode!")
            elif evaluation_mode == "conll" and tagging_scheme:
                raise ValueError("Tagging scheme can be set only in strict mode!")
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
                f"Evaluation mode {evaluation_mode} not supported. Must be one of "
                f"[unit, conll, strict]."
            )

    @property
    def metrics(
        self,
    ) -> Sequence[Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]]]:
        return [self.metric]

    @staticmethod
    def get_metric_name(
        evaluation_mode: EvaluationMode, tagging_scheme: Optional[TaggingScheme] = None
    ) -> str:
        if evaluation_mode == "unit":
            return "UnitSeqeval"
        elif evaluation_mode in SequenceLabelingEvaluator.SEQEVAL_EVALUATION_MODES:
            metric_name = "seqeval"
            if evaluation_mode == "conll":
                metric_name += "__mode_None"  # todo: deal with None in metric names
            else:
                metric_name += "__mode_strict"

            metric_name += f"__scheme_{tagging_scheme}"
            return metric_name
        else:
            raise ValueError(f"Evaluation Mode {evaluation_mode} unsupported.")


EvaluationMode = SequenceLabelingEvaluator.EvaluationMode
TaggingScheme = SequenceLabelingEvaluator.TaggingScheme

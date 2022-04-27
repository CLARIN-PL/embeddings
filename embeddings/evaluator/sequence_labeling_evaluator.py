from dataclasses import asdict
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Sequence, Set, Type, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.evaluation_results import Predictions, SequenceLabelingEvaluationResults
from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.metric import Metric
from embeddings.metric.unit_seqeval_metric import UnitSeqevalMetric


class SequenceLabelingEvaluator(MetricsEvaluator[SequenceLabelingEvaluationResults]):
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
        return_input_data: bool = True,
        evaluation_mode: EvaluationMode = EvaluationMode.CONLL,
        tagging_scheme: Optional[TaggingScheme] = None,
    ) -> None:
        super().__init__(return_input_data)
        self.evaluation_mode = evaluation_mode
        self.tagging_scheme = tagging_scheme

    def _get_metric(
        self,
    ) -> Union[HuggingFaceMetric, UnitSeqevalMetric]:
        if self.evaluation_mode in SequenceLabelingEvaluator.SEQEVAL_EVALUATION_MODES:
            if self.evaluation_mode == "strict" and not self.tagging_scheme:
                raise ValueError("Tagging scheme must be set, when using strict evaluation mode!")
            elif self.evaluation_mode == "conll" and self.tagging_scheme:
                raise ValueError("Tagging scheme can be set only in strict mode!")
            return HuggingFaceMetric(
                name="seqeval",
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

    def metrics(
        self,
    ) -> Sequence[Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]]]:
        return [self._get_metric()]

    @property
    def evaluation_results_cls(self) -> Type[SequenceLabelingEvaluationResults]:
        return SequenceLabelingEvaluationResults

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

    def evaluate(
        self, data: Union[Dict[str, nptyping.NDArray[Any]], Predictions]
    ) -> SequenceLabelingEvaluationResults:
        if isinstance(data, Predictions):
            data = asdict(data)
        result = {}
        [metric] = self.metrics()
        computed = metric.compute(y_true=data["y_true"], y_pred=data["y_pred"])
        result["accuracy"] = computed.pop("overall_accuracy")
        result["f1_micro"] = computed.pop("overall_f1")
        result["recall_micro"] = computed.pop("overall_recall")
        result["precision_micro"] = computed.pop("overall_precision")
        for class_metrics in computed.values():
            class_metrics.pop("number")
        result["data"] = data if self.return_input_data else None
        return self.evaluation_results_cls(**result, classes=computed)


EvaluationMode = SequenceLabelingEvaluator.EvaluationMode
TaggingScheme = SequenceLabelingEvaluator.TaggingScheme

from enum import Enum
from typing import Any, Dict, Final, List, Optional, Set, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.evaluation_results import Predictions, SequenceLabelingEvaluationResults
from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.metric import Metric
from embeddings.metric.seqeval_metric import SeqevalMetric
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

    def metrics(
        self,
    ) -> Dict[str, Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]]]:
        return {
            self.get_metric_name(
                evaluation_mode=self.evaluation_mode, tagging_scheme=self.tagging_scheme
            ): self.get_metric(
                evaluation_mode=self.evaluation_mode, tagging_scheme=self.tagging_scheme
            )
        }

    @staticmethod
    def get_metric(
        evaluation_mode: EvaluationMode, tagging_scheme: Optional[TaggingScheme] = None
    ) -> HuggingFaceMetric:
        if evaluation_mode in SequenceLabelingEvaluator.SEQEVAL_EVALUATION_MODES:
            if evaluation_mode == "strict" and not tagging_scheme:
                raise ValueError("Tagging scheme must be set, when using strict evaluation mode!")
            elif evaluation_mode == "conll" and tagging_scheme:
                raise ValueError("Tagging scheme can be set only in strict mode!")
            return HuggingFaceMetric(
                metric=SeqevalMetric(),
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
        data = Predictions(**data) if isinstance(data, dict) else data
        [metric] = self.metrics().values()
        result = metric.compute(y_true=data.y_true, y_pred=data.y_pred)
        result["data"] = data if self.return_input_data else None
        return SequenceLabelingEvaluationResults(**result)


EvaluationMode = SequenceLabelingEvaluator.EvaluationMode
TaggingScheme = SequenceLabelingEvaluator.TaggingScheme

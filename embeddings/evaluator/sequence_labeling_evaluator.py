from typing import Any, Dict, List, Optional, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.evaluation_results import Predictions, SequenceLabelingEvaluationResults
from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.metric import Metric
from embeddings.metric.sequence_labeling import (
    SEQEVAL_EVALUATION_MODES,
    EvaluationMode,
    TaggingScheme,
    get_sequence_labeling_metric,
)


class SequenceLabelingEvaluator(MetricsEvaluator[SequenceLabelingEvaluationResults]):
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
            ): get_sequence_labeling_metric(
                evaluation_mode=self.evaluation_mode, tagging_scheme=self.tagging_scheme
            )
        }

    @staticmethod
    def get_metric_name(
        evaluation_mode: EvaluationMode, tagging_scheme: Optional[TaggingScheme] = None
    ) -> str:
        if evaluation_mode == "unit":
            return "UnitSeqeval"
        elif evaluation_mode in SEQEVAL_EVALUATION_MODES:
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

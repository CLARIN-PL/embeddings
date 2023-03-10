import abc
from typing import Any, Dict, Generic, List, TypeVar, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.evaluation_results import EvaluationResults, Predictions
from embeddings.evaluator.evaluator import Evaluator
from embeddings.metric.metric import Metric

EvaluationResultsType = TypeVar("EvaluationResultsType", bound=EvaluationResults)


class MetricsEvaluator(
    Evaluator[Predictions, EvaluationResultsType],
    Generic[EvaluationResultsType],
):
    def __init__(self, return_input_data: bool = True):
        self.return_input_data = return_input_data

    @abc.abstractmethod
    def metrics(
        self,
    ) -> Dict[str, Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]]]:
        pass

    @abc.abstractmethod
    def evaluate(
        self, data: Union[Dict[str, nptyping.NDArray[Any]], Predictions]
    ) -> EvaluationResultsType:
        pass

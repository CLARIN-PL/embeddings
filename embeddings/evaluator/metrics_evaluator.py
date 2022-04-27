import abc
from dataclasses import asdict
from typing import Any, Dict, Generic, List, Sequence, Type, TypeVar, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.evaluation_results import EvaluationResults, Predictions
from embeddings.evaluator.evaluator import Evaluator
from embeddings.metric.metric import Metric

EvaluationResultsType = TypeVar("EvaluationResultsType", bound=EvaluationResults)


class MetricsEvaluator(
    Evaluator[Dict[str, nptyping.NDArray[Any]], EvaluationResultsType],
    Generic[EvaluationResultsType],
):
    def __init__(self, return_input_data: bool = True):
        super().__init__()
        self.return_input_data = return_input_data

    @abc.abstractmethod
    def metrics(
        self,
    ) -> Sequence[Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]]]:
        pass

    @property
    @abc.abstractmethod
    def evaluation_results_cls(self) -> Type[EvaluationResultsType]:
        pass

    def evaluate(
        self, data: Union[Dict[str, nptyping.NDArray[Any]], Predictions]
    ) -> EvaluationResultsType:
        if isinstance(data, Predictions):
            data = asdict(data)
        result: Dict[Any, Any] = {
            str(metric): metric.compute(y_true=data["y_true"], y_pred=data["y_pred"])
            for metric in self.metrics()
        }
        a = data if self.return_input_data else None
        result["data"] = a
        return self.evaluation_results_cls(**result)

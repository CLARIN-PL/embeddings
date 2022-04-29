import abc
from typing import Any, Dict, Generic, List, Type, TypeVar, Union

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
        super().__init__()
        self.return_input_data = return_input_data

    @abc.abstractmethod
    def metrics(
        self,
    ) -> Dict[str, Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]]]:
        pass

    @property
    @abc.abstractmethod
    def evaluation_results_cls(self) -> Type[EvaluationResultsType]:
        pass

    @abc.abstractmethod
    def evaluate(
        self, data: Union[Dict[str, nptyping.NDArray[Any]], Predictions]
    ) -> EvaluationResultsType:
        data = Predictions(**data) if isinstance(data, dict) else data
        result: Dict[Any, Any] = {
            name: metric.compute(y_true=data.y_true, y_pred=data.y_pred)
            for name, metric in self.metrics().items()
        }
        a = data if self.return_input_data else None
        result["data"] = a
        return self.evaluation_results_cls(**result)

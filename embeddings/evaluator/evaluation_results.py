from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import srsly
from numpy import typing as nptyping


@dataclass
class Predictions:
    y_pred: nptyping.NDArray[Any]
    y_true: nptyping.NDArray[Any]
    y_probabilities: Optional[nptyping.NDArray[Any]] = None
    names: Optional[nptyping.NDArray[Any]] = None

    def __post_init__(self) -> None:
        if self.y_probabilities and self.names:
            assert self.names.shape[0] == self.y_probabilities.shape[1]

    @staticmethod
    def from_evaluation_json(data: str) -> "Predictions":
        evaluation = srsly.json_loads(data)
        return Predictions(
            y_pred=evaluation["data"]["y_pred"],
            y_true=evaluation["data"]["y_true"],
            y_probabilities=evaluation["data"]["y_probabilities"],
            names=evaluation["data"]["names"],
        )


@dataclass
class EvaluationResults:
    data: Optional[Predictions]

    @property
    def metrics(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("data")
        return result


@dataclass
class TextClassificationEvaluationResults(EvaluationResults):
    accuracy: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    recall_macro: float
    recall_micro: float
    recall_weighted: float
    precision_macro: float
    precision_micro: float
    precision_weighted: float


@dataclass
class SequenceLabelingEvaluationResults(EvaluationResults):
    accuracy: float
    f1_micro: float
    recall_micro: float
    precision_micro: float
    classes: dict[str, dict[str, Union[float, int]]]

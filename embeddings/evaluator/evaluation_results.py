from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import srsly
from numpy import typing as nptyping


@dataclass
class Predictions:
    y_pred: nptyping.NDArray[Any]
    y_true: nptyping.NDArray[Any]
    y_probabilities: Optional[nptyping.NDArray[Any]] = None
    names: Optional[nptyping.NDArray[Any]] = None

    def __post_init__(self) -> None:
        if self.y_probabilities is not None and self.names is not None:
            if len(self.y_probabilities.shape) == 2:  # text classification
                if self.names.shape[0] != self.y_probabilities.shape[1]:
                    raise ValueError("Wrong dimensionality of names and y_probabilities.")
            else:  # sequence labelling
                if self.names.shape[0] != len(self.y_probabilities[0][0]):
                    raise ValueError("Wrong dimensionality of names and y_probabilities.")

    @staticmethod
    def from_evaluation_json(data: str) -> "Predictions":
        evaluation = srsly.json_loads(data)
        dtype = object if isinstance(evaluation["data"]["y_pred"], list) else None
        return Predictions(
            y_pred=np.array(evaluation["data"]["y_pred"], dtype=dtype),
            y_true=np.array(evaluation["data"]["y_true"], dtype=dtype),
            y_probabilities=np.array(evaluation["data"]["y_probabilities"], dtype=dtype),
            names=np.array(evaluation["data"]["names"], dtype=dtype),
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

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import srsly
from numpy import typing as nptyping


class Task(str, Enum):
    sequence_labeling = "sequence_labeling"
    text_classification = "text_classification"


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

    @classmethod
    def from_evaluation_json(cls, data: str) -> "Predictions":
        evaluation = srsly.json_loads(data)
        evaluation_dtype = object if isinstance(evaluation["data"]["y_pred"][0], list) else None
        return cls(
            y_pred=np.array(evaluation["data"]["y_pred"], dtype=evaluation_dtype),
            y_true=np.array(evaluation["data"]["y_true"], dtype=evaluation_dtype),
            y_probabilities=np.array(evaluation["data"]["y_probabilities"], dtype=evaluation_dtype),
            names=np.array(evaluation["data"]["names"], dtype=evaluation_dtype),
        )


Data = Union[Predictions, List[Dict[str, Any]]]


@dataclass
class EvaluationResults:
    def __repr__(self) -> str:
        fields = asdict(self)
        fields.pop("data")
        return (
            self.__class__.__qualname__
            + "("
            + ", ".join([f"{k}={v}" for k, v in fields.items()])
            + ")"
        )

    @property
    def metrics(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("data")
        return result


@dataclass(repr=False)
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
    classes: Dict[str, Dict[str, Union[float, int]]]
    data: Optional[Data] = None


@dataclass(repr=False)
class SequenceLabelingEvaluationResults(EvaluationResults):
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
    classes: Dict[str, Dict[str, Union[float, int]]]
    data: Optional[Data] = None


@dataclass(repr=False)
class QuestionAnsweringEvaluationResults(EvaluationResults):
    exact: float
    f1: float
    total: float
    best_exact: float
    best_exact_thresh: float
    best_f1: float
    best_f1_thresh: float
    HasAns_exact: Optional[float] = None
    HasAns_f1: Optional[float] = None
    HasAns_total: Optional[float] = None
    NoAns_exact: Optional[float] = None
    NoAns_f1: Optional[float] = None
    NoAns_total: Optional[float] = None
    data: Optional[Data] = None

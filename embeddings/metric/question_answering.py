import abc
from typing import List, Dict, Union, Any

import datasets

from embeddings.task.lightning_task.question_answering import SQUAD_V2_PREDICTED_ANSWER_TYPE, SQUAD_V2_GOLD_ANSWER_TYPE


class QAMetric(abc.ABC):
    """
    TODO:
    Refactor pt 2:
    - Decide whether we need additional seperate base QA metric class
    """

    """TODO: Refactor it as metric"""
    @abc.abstractmethod
    def calculate(
        self,
        predictions: List[SQUAD_V2_PREDICTED_ANSWER_TYPE],
        references: List[SQUAD_V2_GOLD_ANSWER_TYPE],
    ) -> Dict[str, Union[float, int]]:
        pass


class SQUADv2Metric(QAMetric):
    """
    TODO:
    Refactor:
    embeddings/metric/question_answering.py
    """

    def __init__(self, no_answer_threshold: float = 1.0) -> None:
        self.metric = datasets.load_metric("squad_v2")
        self.no_answer_threshold = no_answer_threshold

    def calculate(
        self,
        predictions: List[SQUAD_V2_PREDICTED_ANSWER_TYPE],
        references: List[SQUAD_V2_GOLD_ANSWER_TYPE],
    ) -> Dict[Any, Any]:
        metrics = self.metric.compute(
            predictions=predictions,
            references=references,
            no_answer_threshold=self.no_answer_threshold,
        )
        assert metrics is not None
        return metrics
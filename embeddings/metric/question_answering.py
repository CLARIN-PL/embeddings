from typing import List, Dict, Union, Any

import evaluate

from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.model.lightning_module.question_answering import QA_PREDICTED_ANSWER_TYPE, QA_GOLD_ANSWER_TYPE


class SQUADv2Metric(HuggingFaceMetric):
    """
    TODO:
    Refactor:
    embeddings/metric/question_answering.py
    """

    def __init__(self, no_answer_threshold: float = 1.0) -> None:
        self.metric = evaluate.load("squad_v2")
        self.no_answer_threshold = no_answer_threshold

    def calculate(
        self,
        predictions: List[QA_PREDICTED_ANSWER_TYPE],
        references: List[QA_GOLD_ANSWER_TYPE],
    ) -> Dict[Any, Any]:
        metrics = self.metric.compute(
            predictions=predictions,
            references=references,
            no_answer_threshold=self.no_answer_threshold,
        )
        assert metrics is not None
        return metrics

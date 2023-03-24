from typing import Any, Dict, List

import evaluate

from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.model.lightning_module.question_answering import (
    QA_GOLD_ANSWER_TYPE,
    QA_PREDICTED_ANSWER_TYPE,
)


class SQUADv2Metric(HuggingFaceMetric):
    def __init__(self, no_answer_threshold: float = 1.0) -> None:
        self.metric = evaluate.load("squad_v2")
        self.no_answer_threshold = no_answer_threshold

    def calculate(
        self,
        predictions: List[QA_PREDICTED_ANSWER_TYPE],
        references: List[QA_GOLD_ANSWER_TYPE],
    ) -> Dict[Any, Any]:
        expected_predictions = [
            {
                "id": str(i),
                "prediction_text": pred["prediction_text"],
                "no_answer_probability": pred["no_answer_probability"],
            }
            for i, pred in enumerate(predictions)
        ]

        expected_references = [
            {
                "id": str(i),
                "answers": {
                    # mypy thinks that references are not indexable
                    "text": ref["answers"]["text"],  # type: ignore[index]
                    "answer_start": ref["answers"]["answer_start"],  # type: ignore[index]
                },
            }
            for i, ref in enumerate(references)
        ]

        metrics: Dict[Any, Any] = self.metric.compute(
            predictions=expected_predictions,
            references=expected_references,
            no_answer_threshold=self.no_answer_threshold,
        )
        assert metrics is not None
        return metrics

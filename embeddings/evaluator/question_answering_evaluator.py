from typing import Any, Dict, List, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.evaluation_results import Predictions, QuestionAnsweringEvaluationResults
from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.metric import Metric
from embeddings.metric.question_answering import SQUADv2Metric
from embeddings.model.lightning_module.question_answering import (
    QA_GOLD_ANSWER_TYPE,
    QA_PREDICTED_ANSWER_TYPE,
)
from embeddings.transformation.lightning_transformation.question_answering_output_transformation import (
    QAPredictionPostProcessor,
)


class QuestionAnsweringEvaluator(
    MetricsEvaluator[Dict[str, Any], QuestionAnsweringEvaluationResults]
):
    def __init__(self, no_answer_threshold: float = 1.0):
        super().__init__(return_input_data=True)
        self.metric = SQUADv2Metric(no_answer_threshold=no_answer_threshold)
        self.postprocessor = QAPredictionPostProcessor()

    def metrics(
        self,
    ) -> Dict[str, Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]],]:
        return {}

    @staticmethod
    def get_golds_text(references: List[QA_GOLD_ANSWER_TYPE]) -> Union[List[List[str]], List[str]]:
        golds_text = []
        for ref in references:
            answers = ref["answers"]
            assert isinstance(answers, dict)
            golds_text.append(answers["text"])
        return golds_text

    @staticmethod
    def get_predictions_text(predictions: List[QA_PREDICTED_ANSWER_TYPE]) -> List[str]:
        return [str(it["prediction_text"]) for it in predictions]

    def evaluate(
        self, data: Union[Dict[str, nptyping.NDArray[Any]], Predictions, Dict[str, Any]]
    ) -> QuestionAnsweringEvaluationResults:
        assert isinstance(data, dict)
        outputs = self.postprocessor.postprocess(**data)
        references: List[QA_GOLD_ANSWER_TYPE] = []
        for example_id, example in enumerate(outputs):
            references.append(
                {
                    "id": example_id,
                    "answers": {
                        "answer_start": example["answers"]["answer_start"]
                        if example["answers"]
                        else [],
                        "text": example["answers"]["text"] if example["answers"] else [],
                    },
                }
            )
        predictions: List[QA_PREDICTED_ANSWER_TYPE] = [
            {"id": it_id, **it["predicted_answer"]} for it_id, it in enumerate(outputs)
        ]
        metrics = SQUADv2Metric().calculate(predictions=predictions, references=references)
        gold_texts = QuestionAnsweringEvaluator.get_golds_text(references)
        predictions_text = QuestionAnsweringEvaluator.get_predictions_text(predictions)

        return QuestionAnsweringEvaluationResults(
            data=outputs, golds_text=gold_texts, predictions_text=predictions_text, **metrics
        )

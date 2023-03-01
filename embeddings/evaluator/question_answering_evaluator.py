from typing import Any, Dict, List, Tuple, Union

import torch
from numpy import typing as nptyping
from tqdm.auto import tqdm

from embedding.transformation.lightning_transformation.question_answering_output_transformation import (
    QAPredictionPostProcessor,
)
from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.metric import Metric
from embeddings.metric.question_answering import SQUADv2Metric
from embeddings.model.lightning_module.question_answering import (
    QA_GOLD_ANSWER_TYPE,
    QA_PREDICTED_ANSWER_TYPE,
)


class QuestionAnsweringEvaluator(MetricsEvaluator):
    """
    TODO: Correct type hints so that mypy doesn't return any errors
    """

    def metrics(
        self,
    ) -> Dict[str, Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]]]:
        pass

    def __init__(self, no_answer_threshold: float = 1.0):
        self.metric = SQUADv2Metric(no_answer_threshold=no_answer_threshold)
        self.postprocessor = QAPredictionPostProcessor()

    def evaluate(self, scores: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metrics = {}
        outputs = {}

        for split in tqdm(scores.keys(), desc="Split"):
            outputs[split] = self.postprocessor.postprocess(**scores[split])

            references: List[QA_GOLD_ANSWER_TYPE] = []
            for example_id, example in enumerate(outputs[split]):
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
                {"id": it_id, **it["predicted_answer"]} for it_id, it in enumerate(outputs[split])
            ]
            metrics[split] = SQUADv2Metric().calculate(
                predictions=predictions, references=references
            )

        return metrics, outputs

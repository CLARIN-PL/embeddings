import abc
from typing import Dict, Any, Tuple, List

from tqdm.auto import tqdm

from embedding.transformation.lightning_transformation.question_answering_output_transformation import \
    QAPredictionPostProcessor
from embeddings.metric.question_answering import SQUADv2Metric
from embeddings.task.lightning_task.question_answering import SQUAD_V2_GOLD_ANSWER_TYPE, SQUAD_V2_PREDICTED_ANSWER_TYPE


class QAEvaluator(abc.ABC):
    """
    TODO:
    Refactor:
    embeddings/evaluator/qa_evaluator.py
    Refactor pt 2:
    Rewrite it as evaluator
    """
    @abc.abstractmethod
    def evaluate(self, scores: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass


class QASquadV2Evaluator(QAEvaluator):
    """TODO:
    Refactor:
    embeddings/evaluator/qa_evaluator.py
    Refactor pt 2:
    Rewrite it as evaluator"""
    def __init__(self, no_answer_threshold: float = 1.0):
        self.metric = SQUADv2Metric(no_answer_threshold=no_answer_threshold)
        self.postprocessor = QAPredictionPostProcessor()

    def evaluate(self, scores: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        metrics = {}
        outputs = {}

        for split in tqdm(scores.keys(), desc="Split"):
            outputs[split] = self.postprocessor.postprocess(**scores[split])

            references: List[SQUAD_V2_GOLD_ANSWER_TYPE] = []
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
            predictions: List[SQUAD_V2_PREDICTED_ANSWER_TYPE] = [
                {"id": it_id, **it["predicted_answer"]} for it_id, it in enumerate(outputs[split])
            ]
            metrics[split] = SQUADv2Metric().calculate(
                predictions=predictions, references=references
            )

        return metrics, outputs
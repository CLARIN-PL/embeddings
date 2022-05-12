from typing import Any, Dict, List, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.evaluation_results import Predictions, TextClassificationEvaluationResults
from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.metric import Metric
from embeddings.metric.prfs_per_class_metric import PRFSPerClassMetric


class TextClassificationEvaluator(MetricsEvaluator[TextClassificationEvaluationResults]):
    def metrics(
        self,
    ) -> Dict[str, Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]]]:
        return {
            "accuracy": HuggingFaceMetric("accuracy"),
            "f1_macro": HuggingFaceMetric("f1", compute_kwargs={"average": "macro"}),
            "f1_micro": HuggingFaceMetric("f1", compute_kwargs={"average": "micro"}),
            "f1_weighted": HuggingFaceMetric("f1", compute_kwargs={"average": "weighted"}),
            "recall_macro": HuggingFaceMetric("recall", compute_kwargs={"average": "macro"}),
            "recall_micro": HuggingFaceMetric("recall", compute_kwargs={"average": "micro"}),
            "recall_weighted": HuggingFaceMetric("recall", compute_kwargs={"average": "weighted"}),
            "precision_macro": HuggingFaceMetric("precision", compute_kwargs={"average": "macro"}),
            "precision_micro": HuggingFaceMetric("precision", compute_kwargs={"average": "micro"}),
            "precision_weighted": HuggingFaceMetric(
                "precision", compute_kwargs={"average": "weighted"}
            ),
            "classes": PRFSPerClassMetric(),
        }

    def evaluate(
        self, data: Union[Dict[str, nptyping.NDArray[Any]], Predictions]
    ) -> TextClassificationEvaluationResults:
        data = Predictions(**data) if isinstance(data, dict) else data
        result = dict()
        for field, metric in self.metrics().items():
            [computed] = metric.compute(y_true=data.y_true, y_pred=data.y_pred).values()
            result[field] = computed
        result["data"] = data if self.return_input_data else None
        return TextClassificationEvaluationResults(**result)

from typing import Any, Dict, List, Sequence, Type, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.evaluation_results import TextClassificationEvaluationResults
from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.metric import Metric


class TextClassificationEvaluator(MetricsEvaluator[TextClassificationEvaluationResults]):
    @property
    def metrics(
        self,
    ) -> Sequence[Metric[Union[List[Any], nptyping.NDArray[Any], torch.Tensor], Dict[Any, Any]]]:
        return [
            HuggingFaceMetric("accuracy"),
            HuggingFaceMetric("f1", compute_kwargs={"average": "macro"}),
            HuggingFaceMetric("recall", compute_kwargs={"average": "macro"}),
            HuggingFaceMetric("precision", compute_kwargs={"average": "macro"}),
        ]

    @property
    def evaluation_results_cls(self) -> Type[TextClassificationEvaluationResults]:
        return TextClassificationEvaluationResults

from typing import Any, Dict, List, Sequence, Union

import torch
from numpy import typing as nptyping

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.metric import Metric


class TextClassificationEvaluator(MetricsEvaluator):
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

from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch

from embeddings.evaluator.metrics_evaluator import MetricsEvaluator
from embeddings.metric.hugging_face_metric import HuggingFaceMetric
from embeddings.metric.metric import Metric
from embeddings.metric.pos_metric import POSTaggingSeqevalMetric


class SequenceTaggingEvaluator(MetricsEvaluator):
    @property
    def metrics(
        self,
    ) -> Sequence[Metric[Union[List[Any], np.ndarray, torch.Tensor], Dict[Any, Any]]]:
        return [
            HuggingFaceMetric("seqeval"),
        ]


class POSTaggingEvaluator(MetricsEvaluator):
    @property
    def metrics(
        self,
    ) -> Sequence[POSTaggingSeqevalMetric]:
        return [
            POSTaggingSeqevalMetric(),
        ]

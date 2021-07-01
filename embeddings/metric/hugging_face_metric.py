from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
import torch

from embeddings.metric.metric import Metric

HF_metric_input = Union[List[Any], np.ndarray, torch.Tensor]


class HuggingFaceMetric(Metric[HF_metric_input, Dict[Any, Any]]):
    def __init__(
        self, name: str, compute_kwargs: Optional[Dict[str, Any]] = None, **init_kwargs: Any
    ):
        super().__init__(name)
        if init_kwargs.get("process_id", 0) != 0:
            raise ValueError(
                "Metric computation should be run on the main process. "
                "Otherwise it would not return results in dict."
            )

        self.metric = datasets.load_metric(name, **init_kwargs)
        self.compute_kwargs = {} if compute_kwargs is None else compute_kwargs

    def compute(
        self, y_true: Optional[HF_metric_input], y_pred: Optional[HF_metric_input], **kwargs: Any
    ) -> Dict[Any, Any]:
        result = self.metric.compute(
            references=y_true, predictions=y_pred, **self.compute_kwargs, **kwargs
        )
        assert isinstance(result, Dict)
        return result

    def __str__(self) -> str:
        compute_kwargs_str = "__".join(f"{k}_{v}" for k, v in self.compute_kwargs.items())
        if compute_kwargs_str:
            return f"{super().__str__()}__{compute_kwargs_str}"
        else:
            return super().__str__()

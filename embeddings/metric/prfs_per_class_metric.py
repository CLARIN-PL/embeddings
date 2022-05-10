from typing import Any, Dict, List, Optional, Union

import torch
from numpy import typing as nptyping
from sklearn.metrics import precision_recall_fscore_support

from embeddings.metric.metric import Metric

HF_metric_input = Union[List[Any], nptyping.NDArray[Any], torch.Tensor]


class PRFSPerClassMetric(Metric[HF_metric_input, Dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__("PRFSPerClass")

    def compute(
        self,
        y_true: HF_metric_input,
        y_pred: HF_metric_input,
        names: Optional[nptyping.NDArray[Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        p, r, f1, s = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
        class_names = (
            range(len(p)) if names is None else list(names)
        )  # todo: add names to all of the pipelines
        assert len(class_names) == len(p)

        return {
            "classes": {
                class_name: {
                    "precision": p_,
                    "recall": r_,
                    "f1": f1_,
                    "support": s_,
                }
                for class_name, p_, r_, f1_, s_ in zip(class_names, p, r, f1, s)
            }
        }

from itertools import chain
from typing import Dict, Union

import numpy as np
import pytest
from sklearn.metrics import classification_report, precision_recall_fscore_support

from embeddings.evaluator.sequence_tagging_evaluator import POSTaggingEvaluator


@pytest.fixture  # type: ignore
def data() -> Dict[str, np.ndarray]:
    return {
        "y_true": np.array(
            [
                ["VB", "RB", "VB"],
                ["RB", "JJ", "VB", "NN"],
                ["JJ", "RB", "NN", "VB"],
            ]
        ),
        "y_pred": np.array(
            [
                ["RB", "VB", "VB"],
                ["NN", "NN", "VB", "NN"],
                ["JJ", "RB", "RB", "VB"],
            ]
        ),
    }


@pytest.fixture  # type: ignore
def sklearn_metrics(data: Dict[str, np.ndarray]) -> Dict[str, Union[Dict[str, float], float]]:
    out_dict = {}
    out_dict.update(
        classification_report(
            list(chain.from_iterable(data["y_true"])),
            list(chain.from_iterable(data["y_pred"])),
            output_dict=True,
        )
    )
    prfs = precision_recall_fscore_support(
        list(chain.from_iterable(data["y_true"])),
        list(chain.from_iterable(data["y_pred"])),
        average="micro",
    )

    del out_dict["macro avg"]
    del out_dict["weighted avg"]

    accuracy = out_dict.pop("accuracy")
    out_dict = {
        tag: {
            metric.replace("support", "number").replace("f1-score", "f1"): metric_value
            for metric, metric_value in tag_scores.items()
        }
        for tag, tag_scores in out_dict.items()
    }
    out_dict.update(
        {
            "overall_accuracy": accuracy,
            "overall_precision": prfs[0],
            "overall_recall": prfs[1],
            "overall_f1": prfs[2],
        }
    )
    return out_dict


@pytest.fixture  # type: ignore
def seqeval_metrics(data: Dict[str, np.ndarray]) -> Dict[str, Union[Dict[str, float], float]]:
    evaluator = POSTaggingEvaluator()
    out = evaluator.evaluate(data)["POSTaggingSeqevalMetric"]
    assert isinstance(out, dict)
    return out


def test_pos_tagging_metrics(
    sklearn_metrics: Dict[str, Union[Dict[str, float], float]],
    seqeval_metrics: Dict[str, Union[Dict[str, float], float]],
) -> None:
    assert all(
        sklearn_metrics[metric] == pytest.approx(seqeval_metrics[metric])
        for metric in sklearn_metrics.keys()
    )

from itertools import chain
from typing import Any, Dict, Union

import numpy as np
import pytest
from numpy import typing as nptyping
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from embeddings.evaluator.evaluation_results import Predictions, SequenceLabelingEvaluationResults
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator


@pytest.fixture(scope="module")
def data() -> Dict[str, nptyping.NDArray[Any]]:
    return {
        "y_true": np.array(
            [
                ["VB", "RB", "VB", "O"],
                ["RB", "JJ", "VB", "NN"],
                ["JJ", "RB", "NN", "VB"],
            ],
            dtype=object,
        ),
        "y_pred": np.array(
            [
                ["RB", "VB", "VB", "O"],
                ["NN", "NN", "VB", "NN"],
                ["JJ", "RB", "RB", "VB"],
            ],
            dtype=object,
        ),
    }


@pytest.fixture(scope="module")
def ner_data() -> Dict[str, nptyping.NDArray[Any]]:
    return {
        "y_true": np.array(
            [
                ["B-RB", "I-RB", "O"],
                ["B-KT", "O", "B-KT", "O"],
            ],
            dtype=object,
        ),
        "y_pred": np.array(
            [
                ["I-RB", "I-RB", "O"],
                ["B-KT", "O", "I-KT", "O"],
            ],
            dtype=object,
        ),
    }


@pytest.fixture(scope="module")
def sklearn_metrics(
    data: Dict[str, nptyping.NDArray[Any]]
) -> Dict[str, Union[Dict[str, float], float]]:
    out_dict = {}
    out_dict.update(
        classification_report(
            list(filter(lambda tag: tag != "O", chain.from_iterable(data["y_true"]))),
            list(filter(lambda tag: tag != "O", chain.from_iterable(data["y_pred"]))),
            output_dict=True,
        )
    )
    prfs = precision_recall_fscore_support(
        list(filter(lambda tag: tag != "O", chain.from_iterable(data["y_true"]))),
        list(filter(lambda tag: tag != "O", chain.from_iterable(data["y_pred"]))),
        average="micro",
    )

    del out_dict["macro avg"]
    del out_dict["weighted avg"]
    del out_dict["accuracy"]

    accuracy = accuracy_score(
        list(chain.from_iterable(data["y_true"])), list(chain.from_iterable(data["y_pred"]))
    )
    out_dict["classes"] = {
        tag: {
            metric.replace("f1-score", "f1"): metric_value
            for metric, metric_value in tag_scores.items()
            if metric != "support"
        }
        for tag, tag_scores in out_dict.items()
    }
    for k in list(out_dict.keys()):
        if k != "classes":
            out_dict.pop(k)

    out_dict.update(
        {
            "accuracy": accuracy,
            "precision_micro": prfs[0],
            "recall_micro": prfs[1],
            "f1_micro": prfs[2],
        }
    )
    return out_dict


@pytest.fixture(scope="module")
def seqeval_metrics(data: Dict[str, nptyping.NDArray[Any]]) -> SequenceLabelingEvaluationResults:
    evaluator = SequenceLabelingEvaluator(
        evaluation_mode=SequenceLabelingEvaluator.EvaluationMode.UNIT
    )
    out = evaluator.evaluate(data)
    assert isinstance(out, SequenceLabelingEvaluationResults)
    return out


def test_pos_tagging_metrics(
    sklearn_metrics: Dict[str, Union[Dict[str, float], float]],
    seqeval_metrics: SequenceLabelingEvaluationResults,
) -> None:
    assert all(
        sklearn_metrics[metric] == seqeval_metrics.metrics[metric]
        for metric in sklearn_metrics.keys()
    )


def test_conll_metrics(ner_data: Dict[str, nptyping.NDArray[Any]]) -> None:
    evaluator = SequenceLabelingEvaluator(
        evaluation_mode=SequenceLabelingEvaluator.EvaluationMode.CONLL
    )
    out = evaluator.evaluate(ner_data)
    np.testing.assert_almost_equal(out.f1_micro, 1.0)


def test_strict_metrics(ner_data: Dict[str, nptyping.NDArray[Any]]) -> None:
    evaluator = SequenceLabelingEvaluator(
        evaluation_mode=SequenceLabelingEvaluator.EvaluationMode.STRICT,
        tagging_scheme=SequenceLabelingEvaluator.TaggingScheme.IOB2,
    )
    out = evaluator.evaluate(ner_data)
    np.testing.assert_almost_equal(out.f1_micro, 0.5)

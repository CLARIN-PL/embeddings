import itertools
import random
from typing import Dict, List, Optional, Sequence

import datasets
import numpy as np
import pytest
import torch
from datasets import Dataset, Metric

from embeddings.evaluator.evaluation_results import Predictions
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.metric.lightning_seqeval_metric import SeqevalTorchMetric
from embeddings.metric.sequence_labeling import get_sequence_labeling_metric


def get_pseudo_random_y_pred(inputs: List[List[int]], labels: List[List[int]]) -> List[List[int]]:
    random.seed(441)
    for i_pred in inputs:
        for i in range(len(i_pred)):
            if random.random() > 0.8:
                i_pred[i] = random.choice(labels)
    return inputs


def get_predictions(
    dataset: datasets.Dataset, y_predictions: List[List[int]], target_label: str
) -> Predictions:
    y_true = [dataset.features[target_label].feature.int2str(x) for x in dataset[target_label]]
    y_predictions = [dataset.features[target_label].feature.int2str(x) for x in y_predictions]
    return Predictions(y_pred=np.array(y_predictions), y_true=np.array(y_true))


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    dataset = datasets.load_dataset("clarin-pl/kpwr-ner", split="train")
    dataset = dataset.train_test_split(test_size=0.0005, seed=441)
    return dataset["test"]


@pytest.fixture(scope="module")
def dataset_unit() -> Dataset:
    dataset = datasets.load_dataset("clarin-pl/aspectemo", split="train")
    dataset = dataset.train_test_split(test_size=0.0005, seed=441)
    return dataset["test"]


@pytest.fixture(scope="module")
def y_pred(dataset: Dataset) -> List[List[int]]:
    labels = list(set(itertools.chain(*dataset["ner"])))
    return get_pseudo_random_y_pred(inputs=dataset["ner"], labels=labels)


@pytest.fixture(scope="module")
def y_pred_unit(dataset_unit: Dataset) -> List[List[int]]:
    labels = list(set(itertools.chain(*dataset_unit["labels"])))
    return get_pseudo_random_y_pred(inputs=dataset_unit["labels"], labels=labels)


@pytest.fixture(scope="module")
def predictions(dataset: Dataset, y_pred: List[List[int]]) -> Predictions:
    return get_predictions(dataset=dataset, y_predictions=y_pred, target_label="ner")


@pytest.fixture(scope="module")
def predictions_unit(dataset_unit: Dataset, y_pred_unit: List[List[int]]) -> Predictions:
    return get_predictions(dataset=dataset_unit, y_predictions=y_pred_unit, target_label="labels")


@pytest.fixture(scope="module")
def seqeval_computing_kwargs() -> Dict[str, Optional[str]]:
    return {"evaluation_mode": "conll", "tagging_scheme": None}


@pytest.fixture(scope="module")
def seqeval_unit_computing_kwargs() -> Dict[str, Optional[str]]:
    return {"evaluation_mode": "unit", "tagging_scheme": None}


@pytest.fixture(scope="module")
def hf_seqeval_metric(seqeval_computing_kwargs: Dict[str, Optional[str]]) -> Metric:
    return get_sequence_labeling_metric(**seqeval_computing_kwargs)


@pytest.fixture(scope="module")
def hf_seqeval_unit_metric(seqeval_unit_computing_kwargs: Dict[str, Optional[str]]) -> Metric:
    return get_sequence_labeling_metric(**seqeval_unit_computing_kwargs)


@pytest.fixture(scope="module")
def seqeval_torch_metric(
    dataset: Dataset, seqeval_computing_kwargs: Dict[str, Optional[str]]
) -> SeqevalTorchMetric:
    return SeqevalTorchMetric(
        class_label=dataset.features["ner"].feature, average="macro", **seqeval_computing_kwargs
    )


@pytest.fixture(scope="module")
def seqeval_unit_torch_metric(
    dataset_unit: Dataset, seqeval_unit_computing_kwargs: Dict[str, Optional[str]]
) -> SeqevalTorchMetric:
    return SeqevalTorchMetric(
        class_label=dataset_unit.features["labels"].feature,
        average="macro",
        **seqeval_unit_computing_kwargs
    )


@pytest.fixture(scope="module")
def evaluator(seqeval_computing_kwargs: Dict[str, Optional[str]]) -> SequenceLabelingEvaluator:
    return SequenceLabelingEvaluator(**seqeval_computing_kwargs)


@pytest.fixture(scope="module")
def evaluator_unit(
    seqeval_unit_computing_kwargs: Dict[str, Optional[str]]
) -> SequenceLabelingEvaluator:
    return SequenceLabelingEvaluator(**seqeval_unit_computing_kwargs)


def test_example_hf_seqeval(hf_seqeval_metric: Metric, predictions: Predictions):
    results = hf_seqeval_metric.compute(y_pred=predictions.y_pred, y_true=predictions.y_true)
    assert set(results.keys()) == {
        "classes",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "accuracy",
    }

    np.testing.assert_almost_equal(results["f1_micro"], 0.35, decimal=pytest.decimal)  # micro-F1
    np.testing.assert_almost_equal(
        results["classes"]["nam_org_organization"]["f1"], 0.5, decimal=pytest.decimal
    )


def test_example_hf_unit_seqeval(hf_seqeval_unit_metric: Metric, predictions_unit: Predictions):
    results = hf_seqeval_unit_metric.compute(
        y_pred=predictions_unit.y_pred, y_true=predictions_unit.y_true
    )
    assert set(results.keys()) == {
        "classes",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "accuracy",
    }

    np.testing.assert_almost_equal(results["f1_micro"], 0.3870, decimal=pytest.decimal)  # micro-F1
    np.testing.assert_almost_equal(
        results["classes"]["a_minus_m"]["f1"], 0.5, decimal=pytest.decimal
    )


def test_seqeval_metric(
    predictions: Predictions,
    evaluator: SequenceLabelingEvaluator,
):
    results = evaluator.evaluate(data=predictions)
    np.testing.assert_almost_equal(results.metrics["f1_micro"], 0.35, decimal=pytest.decimal)
    np.testing.assert_almost_equal(results.metrics["f1_macro"], 0.34666666, decimal=pytest.decimal)
    assert results.metrics["classes"]["nam_org_organization"]["f1"] == 0.5


def test_unit_seqeval_metric(
    predictions_unit: Predictions,
    evaluator_unit: SequenceLabelingEvaluator,
):
    results = evaluator_unit.evaluate(data=predictions_unit)
    np.testing.assert_almost_equal(results.metrics["f1_micro"], 0.3870, decimal=pytest.decimal)
    np.testing.assert_almost_equal(results.metrics["f1_macro"], 0.3873, decimal=pytest.decimal)
    assert results.metrics["classes"]["a_minus_m"]["f1"] == 0.5


#


def test_lightning_seqeval_metric(
    seqeval_torch_metric: SeqevalTorchMetric, dataset: Dataset, y_pred: List[List[int]]
):
    for preds, targets in zip(y_pred, dataset["ner"]):
        seqeval_torch_metric(torch.tensor(preds), torch.tensor(targets))
    results = seqeval_torch_metric.compute()
    np.testing.assert_almost_equal(results["f1_macro"], 0.34666666, decimal=pytest.decimal)


def test_unit_lightning_seqeval_metric(
    seqeval_unit_torch_metric: SeqevalTorchMetric,
    dataset_unit: Dataset,
    y_pred_unit: List[List[int]],
):
    for preds, targets in zip(y_pred_unit, dataset_unit["labels"]):
        seqeval_unit_torch_metric(torch.tensor(preds), torch.tensor(targets))
    results = seqeval_unit_torch_metric.compute()
    np.testing.assert_almost_equal(results["f1_macro"], 0.3873, decimal=pytest.decimal)

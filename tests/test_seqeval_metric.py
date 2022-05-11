import itertools
import random
from typing import Dict, List, Optional

import datasets
import numpy as np
import pytest
import torch
from datasets import Dataset, Metric

from embeddings.evaluator.evaluation_results import Predictions
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.metric.lightning_seqeval_metric import SeqevalTorchMetric

random.seed(441)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    dataset = datasets.load_dataset("clarin-pl/kpwr-ner", split="train")
    dataset = dataset.train_test_split(test_size=0.0005, seed=441)
    return dataset["test"]


@pytest.fixture(scope="module")
def y_pred(dataset: Dataset) -> List[List[int]]:
    labels = list(set(itertools.chain(*dataset["ner"])))
    preds = dataset["ner"]
    for i_pred in preds:
        for i in range(len(i_pred)):
            if random.random() > 0.8:
                i_pred[i] = random.choice(labels)
    return preds


@pytest.fixture(scope="module")
def predictions(dataset: Dataset, y_pred: List[List[int]]) -> Predictions:
    y_true = [dataset.features["ner"].feature.int2str(x) for x in dataset["ner"]]
    y_pred = [dataset.features["ner"].feature.int2str(x) for x in y_pred]
    return Predictions(y_pred=np.array(y_pred), y_true=np.array(y_true))


@pytest.fixture(scope="module")
def seqeval_computing_kwargs() -> Dict[str, Optional[str]]:
    return {"evaluation_mode": "conll", "tagging_scheme": None}


@pytest.fixture(scope="module")
def hf_seqeval_metric(seqeval_computing_kwargs: Dict[str, Optional[str]]) -> Metric:
    return datasets.load_metric("seqeval", **seqeval_computing_kwargs)


@pytest.fixture(scope="module")
def seqeval_torch_metric(
    dataset: Dataset, seqeval_computing_kwargs: Dict[str, Optional[str]]
) -> SeqevalTorchMetric:
    return SeqevalTorchMetric(
        class_label=dataset.features["ner"].feature,
        metric_name="f1_macro",
        **seqeval_computing_kwargs
    )


@pytest.fixture(scope="module")
def evaluator(seqeval_computing_kwargs: Dict[str, Optional[str]]) -> SequenceLabelingEvaluator:
    return SequenceLabelingEvaluator(**seqeval_computing_kwargs)


def test_examplar(hf_seqeval_metric: Metric, predictions: Predictions):
    results = hf_seqeval_metric.compute(
        predictions=predictions.y_pred, references=predictions.y_true
    )
    assert list(results.keys()) == [
        "nam_adj_country",
        "nam_eve_human_sport",
        "nam_liv_god",
        "nam_liv_person",
        "nam_loc_gpe_city",
        "nam_loc_gpe_country",
        "nam_org_group_band",
        "nam_org_institution",
        "nam_org_organization",
        "nam_pro_title_album",
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "overall_accuracy",
    ]
    np.testing.assert_almost_equal(results["overall_f1"], 0.35, decimal=pytest.decimal)  # micro-F1
    assert results["nam_org_organization"]["f1"] == 0.5


def test_seqeval_metric(
    predictions: Predictions,
    evaluator: SequenceLabelingEvaluator,
):
    results = evaluator.evaluate(data=predictions)
    np.testing.assert_almost_equal(results.metrics["f1_micro"], 0.35, decimal=pytest.decimal)
    np.testing.assert_almost_equal(results.metrics["f1_macro"], 0.34666666, decimal=pytest.decimal)
    assert results.metrics["classes"]["nam_org_organization"]["f1"] == 0.5


def test_lightning_seqeval_metric(
    seqeval_torch_metric: SeqevalTorchMetric, dataset: Dataset, y_pred: List[List[int]]
):
    for preds, targets in zip(y_pred, dataset["ner"]):
        seqeval_torch_metric(torch.tensor(preds), torch.tensor(targets))
    f1_macro = seqeval_torch_metric.compute()
    np.testing.assert_almost_equal(f1_macro, 0.34666666, decimal=pytest.decimal)

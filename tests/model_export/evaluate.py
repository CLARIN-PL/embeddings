from pathlib import Path
from typing import List

import numpy
import torch
from onnxruntime import InferenceSession
from transformers import AutoModel

from embeddings.data.datamodule import TextClassificationDataModule
from embeddings.evaluator.evaluation_results import Predictions, TextClassificationEvaluationResults
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator


def evaluate_hf_model_text_classification(
    model: AutoModel, datamodule: TextClassificationDataModule
) -> TextClassificationEvaluationResults:
    preds = []
    y_true = []
    for batch in datamodule.get_subset(subset="test"):
        preds += torch.argmax(model(**batch).logits, dim=1)
        y_true += batch["labels"]

    y_pred = torch.IntTensor(preds).numpy()
    y_true = torch.IntTensor(y_true).numpy()
    predictions = Predictions(y_pred=y_pred, y_true=y_true)
    return TextClassificationEvaluator().evaluate(predictions)


def evaluate_onnx_text_classification(
    model_path: Path, datamodule: TextClassificationDataModule
) -> TextClassificationEvaluationResults:
    session = InferenceSession(str(model_path / "model.onnx"))

    preds = []
    y_true = []
    for batch in datamodule.get_subset("test"):
        np_batch = {
            "input_ids": batch["input_ids"].numpy(),
            "token_type_ids": batch["token_type_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
        }
        outputs = session.run(output_names=["logits"], input_feed=np_batch)[0]
        preds += torch.argmax(torch.Tensor(outputs), dim=1)
        y_true += batch["labels"]

    y_pred = torch.IntTensor(preds).numpy()
    y_true = torch.IntTensor(y_true).numpy()
    predictions = Predictions(y_pred=y_pred, y_true=y_true)
    return TextClassificationEvaluator().evaluate(predictions)


def assert_metrics_almost_equal(
    pretrained_model_metrics: TextClassificationEvaluationResults,
    loaded_model_metrics: TextClassificationEvaluationResults,
    metric_keys: List[str],
    decimal: float,
):
    for metric_key in metric_keys:
        numpy.testing.assert_almost_equal(
            getattr(pretrained_model_metrics, metric_key),
            getattr(loaded_model_metrics, metric_key),
            decimal=decimal,
        )

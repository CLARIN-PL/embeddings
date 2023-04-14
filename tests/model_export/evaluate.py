import json
from pathlib import Path
from typing import List, Union

import numpy
import numpy as np
import torch
from onnxruntime import InferenceSession
from transformers import AutoModel

from embeddings.data.datamodule import SequenceLabelingDataModule, TextClassificationDataModule
from embeddings.data.qa_datamodule import QuestionAnsweringDataModule
from embeddings.evaluator.evaluation_results import (
    Predictions,
    QuestionAnsweringEvaluationResults,
    SequenceLabelingEvaluationResults,
    TextClassificationEvaluationResults,
)
from embeddings.evaluator.question_answering_evaluator import QuestionAnsweringEvaluator
from embeddings.evaluator.sequence_labeling_evaluator import SequenceLabelingEvaluator
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.task.lightning_task.question_answering import QuestionAnsweringTask


def assert_metrics_almost_equal(
    pretrained_model_metrics: Union[
        TextClassificationEvaluationResults,
        SequenceLabelingEvaluationResults,
        QuestionAnsweringEvaluationResults,
    ],
    loaded_model_metrics: Union[
        TextClassificationEvaluationResults,
        SequenceLabelingEvaluationResults,
        QuestionAnsweringEvaluationResults,
    ],
    metric_keys: List[str],
    decimal: float,
):
    for metric_key in metric_keys:
        numpy.testing.assert_almost_equal(
            getattr(pretrained_model_metrics, metric_key),
            getattr(loaded_model_metrics, metric_key),
            decimal=decimal,
        )


def evaluate_hf_model_text_classification(
    model: AutoModel, datamodule: TextClassificationDataModule
) -> TextClassificationEvaluationResults:
    preds = []
    y_true = []

    with torch.no_grad():
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


def evaluate_hf_model_token_classification(
    model: AutoModel, datamodule: SequenceLabelingDataModule
) -> SequenceLabelingEvaluationResults:
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in datamodule.get_subset(subset="test"):
            y_pred += torch.argmax(model(**batch).logits, dim=2)
            y_true += batch["labels"]

    for i, (pred, gt) in enumerate(zip(y_pred, y_true)):
        y_pred[i] = [model.config.id2label[x.item()] for x in pred[gt != -100]]
        y_true[i] = [model.config.id2label[x.item()] for x in gt[gt != -100]]
        assert len(y_pred[i]) == len(y_true[i])

    predictions = Predictions(
        y_pred=np.array(y_pred, dtype=object), y_true=np.array(y_true, dtype=object)
    )
    return SequenceLabelingEvaluator().evaluate(predictions)


def evaluate_onnx_token_classification(
    model_path: Path, datamodule: SequenceLabelingDataModule
) -> SequenceLabelingEvaluationResults:
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
    session = InferenceSession(str(model_path / "model.onnx"))

    y_pred = []
    y_true = []
    for batch in datamodule.get_subset("test"):
        np_batch = {
            "input_ids": batch["input_ids"].numpy(),
            "token_type_ids": batch["token_type_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
        }
        outputs = session.run(output_names=["logits"], input_feed=np_batch)[0]
        y_pred += torch.argmax(torch.Tensor(outputs), dim=2)
        y_true += batch["labels"]

    #     import pdb; pdb.set_trace()
    for i, (pred, gt) in enumerate(zip(y_pred, y_true)):
        y_pred[i] = [config["id2label"][str(x.item())] for x in pred[gt != -100]]
        y_true[i] = [config["id2label"][str(x.item())] for x in gt[gt != -100]]

    predictions = Predictions(
        y_pred=np.array(y_pred, dtype=object), y_true=np.array(y_true, dtype=object)
    )
    return SequenceLabelingEvaluator().evaluate(predictions)


def evaluate_hf_model_question_answering(
    model: AutoModel,
    datamodule: QuestionAnsweringDataModule,
) -> QuestionAnsweringEvaluationResults:
    with torch.no_grad():
        outputs = []
        for batch in datamodule.get_subset(subset="test"):
            outputs.append({"data": batch, "outputs": dict(model.forward(**batch).items())})

    outputs_processed = QuestionAnsweringTask.postprocess_outputs(
        outputs, data=datamodule, predict_subset="test"
    )
    return QuestionAnsweringEvaluator().evaluate(data=outputs_processed)


def evaluate_onnx_model_question_answering(
    model_path: Path, datamodule: QuestionAnsweringDataModule
) -> QuestionAnsweringEvaluationResults:
    session = InferenceSession(str(model_path / "model.onnx"))

    outputs = []
    for batch in datamodule.get_subset("test"):
        np_batch = {
            "input_ids": batch["input_ids"].numpy(),
            "token_type_ids": batch["token_type_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
        }
        model_outputs = session.run(
            output_names=["start_logits", "end_logits"], input_feed=np_batch
        )
        outputs.append(
            {
                "data": batch,
                "outputs": {"start_logits": model_outputs[0], "end_logits": model_outputs[1]},
            }
        )

    outputs_processed = QuestionAnsweringTask.postprocess_outputs(
        outputs, data=datamodule, predict_subset="test"
    )
    return QuestionAnsweringEvaluator().evaluate(data=outputs_processed)

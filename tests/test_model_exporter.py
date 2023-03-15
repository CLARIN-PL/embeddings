from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy.testing
import pytest
import torch
from _pytest.tmpdir import TempdirFactory
from onnxruntime import InferenceSession
from transformers import AlbertForSequenceClassification

from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.data.datamodule import TextClassificationDataModule
from embeddings.evaluator.evaluation_results import Predictions, TextClassificationEvaluationResults
from embeddings.evaluator.text_classification_evaluator import TextClassificationEvaluator
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.utils.model_exporter import HuggingFaceModelExporter, ONNXModelExporter

LIGHTNING_TEXT_CLASSIFICATION_PIPELINE_OUTPUT_TYPE = Tuple[
    LightningClassificationPipeline, TextClassificationEvaluationResults
]


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def comparison_metrics_keys() -> List[str]:
    return [
        "accuracy",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
    ]


@pytest.fixture(scope="module")
def embedding_name() -> str:
    return "hf-internal-testing/tiny-albert"


@pytest.fixture(scope="module")
def dataset_kwargs(tmp_path_module: Path) -> Dict[str, Any]:
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/polemo2-official",
        load_dataset_kwargs={
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
        persist_path=str(tmp_path_module / "data"),
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": tmp_path_module / "data",
        "input_column_name": ["text"],
        "target_column_name": "target",
    }


@pytest.fixture(scope="module")
def config() -> LightningAdvancedConfig:
    return LightningAdvancedConfig(
        finetune_last_n_layers=0,
        task_train_kwargs={
            "max_epochs": 1,
            "deterministic": True,
        },
        task_model_kwargs={
            "learning_rate": 5e-4,
            "train_batch_size": 32,
            "eval_batch_size": 32,
            "use_scheduler": True,
            "optimizer": "AdamW",
            "adam_epsilon": 1e-8,
            "warmup_steps": 100,
            "weight_decay": 0.0,
        },
        datamodule_kwargs={
            "max_seq_length": 64,
        },
        early_stopping_kwargs={},
        tokenizer_kwargs={},
        batch_encoding_kwargs={},
        dataloader_kwargs={},
        model_config_kwargs={},
    )


@pytest.fixture(scope="module")
def lightning_text_classification_pipeline(
    dataset_kwargs: Dict[str, Any],
    config: LightningAdvancedConfig,
    embedding_name: str,
    tmp_path_module: Path,
) -> LIGHTNING_TEXT_CLASSIFICATION_PIPELINE_OUTPUT_TYPE:
    pipeline = LightningClassificationPipeline(
        embedding_name_or_path=embedding_name,
        devices="auto",
        accelerator="cpu",
        output_path=".",
        config=config,
        **dataset_kwargs
    )
    metrics = pipeline.run()
    return pipeline, metrics


@pytest.fixture(scope="module")
def hf_datamodule(
    embedding_name: str, dataset_kwargs: Dict[str, Any], config: LightningAdvancedConfig
) -> TextClassificationDataModule:
    return TextClassificationDataModule(
        tokenizer_name_or_path=embedding_name,
        dataset_name_or_path=dataset_kwargs["dataset_name_or_path"],
        text_fields=dataset_kwargs["input_column_name"],
        target_field=dataset_kwargs["target_column_name"],
        train_batch_size=config.task_model_kwargs["train_batch_size"],
        eval_batch_size=config.task_model_kwargs["eval_batch_size"],
        tokenizer_kwargs=config.tokenizer_kwargs,
        batch_encoding_kwargs=config.batch_encoding_kwargs,
        load_dataset_kwargs={},
        dataloader_kwargs=config.dataloader_kwargs,
        **config.datamodule_kwargs
    )


def evaluate_hf_model(
    model_path: Path, datamodule: TextClassificationDataModule
) -> TextClassificationEvaluationResults:
    model = AlbertForSequenceClassification.from_pretrained(model_path)
    preds = []
    y_true = []
    for batch in datamodule.get_subset(subset="test"):
        preds += torch.argmax(model(**batch).logits, dim=1)
        y_true += batch["labels"]

    y_pred = torch.IntTensor(preds).numpy()
    y_true = torch.IntTensor(y_true).numpy()
    predictions = Predictions(y_pred=y_pred, y_true=y_true)
    return TextClassificationEvaluator().evaluate(predictions)


def evaluate_onnx_model(
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
) -> None:
    for metric_key in metric_keys:
        numpy.testing.assert_almost_equal(
            getattr(pretrained_model_metrics, metric_key),
            getattr(loaded_model_metrics, metric_key),
            decimal=pytest.decimal,
        )


def test_hf_model_exporter_from_pipeline(
    embedding_name: str,
    lightning_text_classification_pipeline: LIGHTNING_TEXT_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    dataset_kwargs: Dict[str, Any],
    comparison_metrics_keys: List[str],
    hf_datamodule: TextClassificationDataModule,
):
    path = tmp_path_module / "hf_model_pipeline"
    exporter = HuggingFaceModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_text_classification_pipeline
    exporter.export_model_from_pipeline(pretrained_pipeline)

    loaded_model_results = evaluate_hf_model(model_path=path, datamodule=hf_datamodule)
    assert_metrics_almost_equal(
        pretrained_model_metrics=pretrained_metrics,
        loaded_model_metrics=loaded_model_results,
        metric_keys=comparison_metrics_keys,
    )


def test_hf_model_exporter_from_task(
    embedding_name: str,
    lightning_text_classification_pipeline: LIGHTNING_TEXT_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    dataset_kwargs: Dict[str, Any],
    comparison_metrics_keys: List[str],
    hf_datamodule: TextClassificationDataModule,
):
    path = tmp_path_module / "hf_model_task"
    exporter = HuggingFaceModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_text_classification_pipeline
    exporter.export_model_from_task(pretrained_pipeline.model.task)

    loaded_model_results = evaluate_hf_model(model_path=path, datamodule=hf_datamodule)
    assert_metrics_almost_equal(
        pretrained_model_metrics=pretrained_metrics,
        loaded_model_metrics=loaded_model_results,
        metric_keys=comparison_metrics_keys,
    )


def test_onnx_model_exporter_from_pipeline(
    embedding_name: str,
    lightning_text_classification_pipeline: LIGHTNING_TEXT_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    dataset_kwargs: Dict[str, Any],
    comparison_metrics_keys: List[str],
    hf_datamodule: TextClassificationDataModule,
):
    path = tmp_path_module / "onnx_model_pipeline"
    exporter = ONNXModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_text_classification_pipeline
    exporter.export_model_from_pipeline(pretrained_pipeline)

    loaded_model_results = evaluate_onnx_model(model_path=path, datamodule=hf_datamodule)
    assert_metrics_almost_equal(
        pretrained_model_metrics=pretrained_metrics,
        loaded_model_metrics=loaded_model_results,
        metric_keys=comparison_metrics_keys,
    )


def test_onnx_model_exporter_from_task(
    embedding_name: str,
    lightning_text_classification_pipeline: LIGHTNING_TEXT_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    dataset_kwargs: Dict[str, Any],
    comparison_metrics_keys: List[str],
    hf_datamodule: TextClassificationDataModule,
):
    path = tmp_path_module / "onnx_model_task"
    exporter = ONNXModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_text_classification_pipeline
    exporter.export_model_from_task(pretrained_pipeline.model.task)

    loaded_model_results = evaluate_onnx_model(model_path=path, datamodule=hf_datamodule)
    assert_metrics_almost_equal(
        pretrained_model_metrics=pretrained_metrics,
        loaded_model_metrics=loaded_model_results,
        metric_keys=comparison_metrics_keys,
    )

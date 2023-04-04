from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from _pytest.tmpdir import TempdirFactory
from transformers import AlbertForTokenClassification

from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.data.datamodule import SequenceLabelingDataModule
from embeddings.evaluator.evaluation_results import SequenceLabelingEvaluationResults
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_sequence_labeling import LightningSequenceLabelingPipeline
from embeddings.task.lightning_task.sequence_labeling import SequenceLabelingTask
from embeddings.utils.model_exporter import HuggingFaceModelExporter, ONNXModelExporter
from tests.model_export.evaluate import (
    assert_metrics_almost_equal,
    evaluate_hf_model_token_classification,
    evaluate_onnx_token_classification,
)

LIGHTNING_TOKEN_CLASSIFICATION_PIPELINE_OUTPUT_TYPE = Tuple[
    LightningSequenceLabelingPipeline, SequenceLabelingEvaluationResults
]


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def comparison_metrics_keys() -> List[str]:
    return [
        "accuracy",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "recall_macro",
        "recall_micro",
        "recall_weighted",
        "precision_macro",
        "precision_micro",
        "precision_weighted",
    ]


@pytest.fixture(scope="module")
def embedding_name() -> str:
    return "hf-internal-testing/tiny-albert"


@pytest.fixture(scope="module")
def dataset_kwargs(tmp_path_module: Path) -> Dict[str, Any]:
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/kpwr-ner",
        load_dataset_kwargs=None,
        persist_path=str(tmp_path_module),
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": tmp_path_module,
        "input_column_name": "tokens",
        "target_column_name": "ner",
    }


@pytest.fixture(scope="module")
def config() -> LightningAdvancedConfig:
    return LightningAdvancedConfig(
        finetune_last_n_layers=0,
        task_train_kwargs={
            "max_epochs": 1,
            "devices": "auto",
            "accelerator": "cpu",
            "deterministic": True,
        },
        task_model_kwargs={
            "learning_rate": 1e-4,
            "train_batch_size": 32,
            "eval_batch_size": 32,
            "use_scheduler": False,
            "optimizer": "AdamW",
            "adam_epsilon": 1e-8,
            "warmup_steps": 100,
            "weight_decay": 0.0,
        },
        datamodule_kwargs={
            "max_seq_length": 64,
        },
        early_stopping_kwargs={
            "monitor": "val/Loss",
            "mode": "min",
            "patience": 3,
        },
        tokenizer_kwargs={},
        batch_encoding_kwargs={},
        dataloader_kwargs={},
        model_config_kwargs={},
    )


@pytest.fixture(scope="module")
def hf_datamodule(
    embedding_name: str, dataset_kwargs: Dict[str, Any], config: LightningAdvancedConfig
) -> SequenceLabelingDataModule:
    return SequenceLabelingDataModule(
        tokenizer_name_or_path=embedding_name,
        dataset_name_or_path=dataset_kwargs["dataset_name_or_path"],
        text_field=dataset_kwargs["input_column_name"],
        target_field=dataset_kwargs["target_column_name"],
        train_batch_size=config.task_model_kwargs["train_batch_size"],
        eval_batch_size=config.task_model_kwargs["eval_batch_size"],
        tokenizer_kwargs=config.tokenizer_kwargs,
        batch_encoding_kwargs=config.batch_encoding_kwargs,
        load_dataset_kwargs={},
        dataloader_kwargs=config.dataloader_kwargs,
        **config.datamodule_kwargs,
    )


@pytest.fixture
def lightning_sequence_labeling_pipeline(
    dataset_kwargs: Dict[str, Any],
    embedding_name: str,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
) -> LIGHTNING_TOKEN_CLASSIFICATION_PIPELINE_OUTPUT_TYPE:
    pl.seed_everything(441)
    pipeline = LightningSequenceLabelingPipeline(
        output_path=tmp_path_module,
        embedding_name_or_path=embedding_name,
        config=config,
        model_checkpoint_kwargs={
            "filename": "last",
            "monitor": None,
            "save_last": False,
        },
        **dataset_kwargs,
    )
    result = pipeline.run()
    return pipeline, result


def test_lightning_sequence_labeling_task_name(
    lightning_sequence_labeling_pipeline: LIGHTNING_TOKEN_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
):
    assert lightning_sequence_labeling_pipeline[0].model.task.hf_task_name.value == "token-classification"


def test_lightning_sequence_labeling_pipeline(
    lightning_sequence_labeling_pipeline: LIGHTNING_TOKEN_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
    tmp_path_module: Path,
) -> None:
    pipeline, result = lightning_sequence_labeling_pipeline

    assert_result_values(result)
    assert_result_types(result)
    assert_inference_from_checkpoint(result, pipeline, tmp_path_module)


def test_hf_model_exporter_from_pipeline(
    embedding_name: str,
    lightning_sequence_labeling_pipeline: LIGHTNING_TOKEN_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    dataset_kwargs: Dict[str, Any],
    comparison_metrics_keys: List[str],
    hf_datamodule: SequenceLabelingDataModule,
):
    path = tmp_path_module / "hf_model_pipeline"
    exporter = HuggingFaceModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_sequence_labeling_pipeline
    exporter.export_model_from_pipeline(pretrained_pipeline)

    loaded_model = AlbertForTokenClassification.from_pretrained(path)
    loaded_model_results = evaluate_hf_model_token_classification(
        model=loaded_model, datamodule=hf_datamodule
    )
    assert_metrics_almost_equal(
        pretrained_model_metrics=pretrained_metrics,
        loaded_model_metrics=loaded_model_results,
        metric_keys=comparison_metrics_keys,
        decimal=pytest.decimal,
    )


def test_hf_model_exporter_from_task(
    embedding_name: str,
    lightning_sequence_labeling_pipeline: LIGHTNING_TOKEN_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    dataset_kwargs: Dict[str, Any],
    comparison_metrics_keys: List[str],
    hf_datamodule: SequenceLabelingDataModule,
):
    path = tmp_path_module / "hf_model_task"
    exporter = HuggingFaceModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_sequence_labeling_pipeline
    exporter.export_model_from_task(pretrained_pipeline.model.task)

    loaded_model = AlbertForTokenClassification.from_pretrained(path)
    loaded_model_results = evaluate_hf_model_token_classification(
        model=loaded_model, datamodule=hf_datamodule
    )
    assert_metrics_almost_equal(
        pretrained_model_metrics=pretrained_metrics,
        loaded_model_metrics=loaded_model_results,
        metric_keys=comparison_metrics_keys,
        decimal=pytest.decimal,
    )


def test_onnx_model_exporter_from_pipeline(
    embedding_name: str,
    lightning_sequence_labeling_pipeline: LIGHTNING_TOKEN_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    dataset_kwargs: Dict[str, Any],
    comparison_metrics_keys: List[str],
    hf_datamodule: SequenceLabelingDataModule,
):
    path = tmp_path_module / "onnx_model_pipeline"
    exporter = ONNXModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_sequence_labeling_pipeline
    exporter.export_model_from_pipeline(pretrained_pipeline)

    loaded_model_results = evaluate_onnx_token_classification(
        model_path=path, datamodule=hf_datamodule
    )
    assert_metrics_almost_equal(
        pretrained_model_metrics=pretrained_metrics,
        loaded_model_metrics=loaded_model_results,
        metric_keys=comparison_metrics_keys,
        decimal=pytest.decimal,
    )


def test_onnx_model_exporter_from_task(
    embedding_name: str,
    lightning_sequence_labeling_pipeline: LIGHTNING_TOKEN_CLASSIFICATION_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    dataset_kwargs: Dict[str, Any],
    comparison_metrics_keys: List[str],
    hf_datamodule: SequenceLabelingDataModule,
):
    path = tmp_path_module / "onnx_model_task"
    exporter = ONNXModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_sequence_labeling_pipeline
    exporter.export_model_from_task(pretrained_pipeline.model.task)

    loaded_model_results = evaluate_onnx_token_classification(
        model_path=path, datamodule=hf_datamodule
    )
    assert_metrics_almost_equal(
        pretrained_model_metrics=pretrained_metrics,
        loaded_model_metrics=loaded_model_results,
        metric_keys=comparison_metrics_keys,
        decimal=pytest.decimal,
    )


def assert_result_values(result: SequenceLabelingEvaluationResults) -> None:
    np.testing.assert_almost_equal(
        result.accuracy,
        0.0019653,
        decimal=pytest.decimal,
    )
    np.testing.assert_almost_equal(result.f1_micro, 0.003776, decimal=pytest.decimal)
    np.testing.assert_almost_equal(
        result.precision_micro,
        0.0020101,
        decimal=pytest.decimal,
    )
    np.testing.assert_almost_equal(
        result.recall_micro,
        0.0310881,
        decimal=pytest.decimal,
    )


def assert_result_types(result: SequenceLabelingEvaluationResults) -> None:
    assert isinstance(result.data.y_true, np.ndarray)
    assert isinstance(result.data.y_probabilities, np.ndarray)
    assert isinstance(result.data.y_pred, np.ndarray)
    assert isinstance(result.data.names, np.ndarray)
    assert isinstance(result.data.y_pred[0], list)
    assert isinstance(result.data.y_true[0], list)
    assert isinstance(result.data.y_probabilities[0], list)
    assert isinstance(result.data.names, np.ndarray)
    assert isinstance(result.data.y_pred[0][0], str)
    assert isinstance(result.data.y_true[0][0], str)
    assert isinstance(result.data.y_probabilities[0][0], np.ndarray)
    assert isinstance(result.data.names[0], str)
    assert isinstance(result.data.y_probabilities[0][0][0], np.float32)


def assert_inference_from_checkpoint(
    result: Dict[str, Any],
    pipeline: LightningSequenceLabelingPipeline,
    tmp_path_module: Path,
) -> None:
    ckpt_path = tmp_path_module / "checkpoints" / "last.ckpt"
    task_from_ckpt = SequenceLabelingTask.from_checkpoint(
        checkpoint_path=ckpt_path.resolve(),
        output_path=tmp_path_module,
    )

    model_state_dict = pipeline.model.task.model.model.state_dict()
    model_from_ckpt_state_dict = task_from_ckpt.model.model.state_dict()
    assert model_state_dict.keys() == model_from_ckpt_state_dict.keys()
    for k in model_state_dict.keys():
        assert torch.equal(model_state_dict[k], model_from_ckpt_state_dict[k])

    predictions = task_from_ckpt.predict(pipeline.datamodule.test_dataloader())
    np.testing.assert_almost_equal(
        result.data.y_probabilities[0][0], predictions.y_probabilities[0][0]
    )

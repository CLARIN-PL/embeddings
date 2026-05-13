from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import lightning as L
import numpy as np
import pytest
from _pytest.tmpdir import TempdirFactory
from transformers import AlbertForQuestionAnswering

from embeddings.config.lightning_config import LightningAdvancedConfig, LightningQABasicConfig
from embeddings.data.qa_datamodule import QuestionAnsweringDataModule
from embeddings.evaluator.evaluation_results import QuestionAnsweringEvaluationResults
from embeddings.pipeline.lightning_question_answering import LightningQuestionAnsweringPipeline
from embeddings.utils.model_exporter import HuggingFaceModelExporter, ONNXModelExporter
from tests.fixtures.sample_qa_dataset import sample_question_answering_dataset
from tests.model_export.evaluate import (
    assert_metrics_almost_equal,
    evaluate_hf_model_question_answering,
    evaluate_onnx_model_question_answering,
)

LIGHTNING_QA_PIPELINE_OUTPUT_TYPE = Tuple[
    LightningQuestionAnsweringPipeline, QuestionAnsweringEvaluationResults
]


@pytest.fixture(scope="module")
def comparison_metrics_keys() -> List[str]:
    return [
        "exact",
        "f1",
        "total",
        "best_exact",
        "best_exact_thresh",
        "best_f1",
        "best_f1_thresh",
        "HasAns_exact",
        "HasAns_f1",
        "HasAns_total",
    ]


@pytest.fixture(scope="module")
def embedding_name() -> str:
    return "hf-internal-testing/tiny-albert"


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def qa_dataset_path(
    tmp_path_module: Path, sample_question_answering_dataset: datasets.DatasetDict
) -> Path:
    dataset = sample_question_answering_dataset
    ds_path = tmp_path_module / "data_sample"
    dataset.save_to_disk(ds_path)
    return ds_path


@pytest.fixture(scope="module")
def dataset_kwargs(qa_dataset_path: Path) -> Dict[str, str]:
    return {
        "dataset_name_or_path": str(qa_dataset_path),
        "context_column_name": "context",
        "question_column_name": "question",
        "answer_column_name": "answers",
    }


@pytest.fixture(scope="module")
def config() -> LightningAdvancedConfig:
    return LightningAdvancedConfig(
        finetune_last_n_layers=0,
        task_train_kwargs={
            "max_epochs": 1,
            "deterministic": True,
            "devices": "auto",
            "accelerator": "cpu",
        },
        task_model_kwargs={
            "learning_rate": 5e-4,
            "train_batch_size": 32,
            "eval_batch_size": 32,
            "use_scheduler": False,
            "optimizer": "Adam",
            "adam_epsilon": 1e-8,
            "warmup_steps": None,
            "weight_decay": 1e-3,
            "doc_stride": 64,
        },
        datamodule_kwargs={
            "max_seq_length": 128,
        },
        early_stopping_kwargs={},
        tokenizer_kwargs={},
        batch_encoding_kwargs={
            "padding": "max_length",
            "truncation": "only_second",
            "return_offsets_mapping": True,
            "return_overflowing_tokens": True,
        },
        dataloader_kwargs={},
        model_config_kwargs={},
    )


@pytest.fixture(scope="module")
def basic_config() -> LightningQABasicConfig:
    return LightningQABasicConfig(
        finetune_last_n_layers=1,
        max_epochs=1,
        learning_rate=5e-4,
        use_scheduler=False,
        optimizer="Adam",
        adam_epsilon=1e-8,
        warmup_steps=None,
        weight_decay=1e-3,
        max_seq_length=128,
        doc_stride=64,
        early_stopping_monitor="val/Loss",
        early_stopping_mode="min",
        early_stopping_patience=3,
    )


@pytest.fixture(scope="module")
def hf_datamodule(
    embedding_name: str, dataset_kwargs: Dict[str, str], config: LightningAdvancedConfig
) -> QuestionAnsweringDataModule:
    dm = QuestionAnsweringDataModule(
        dataset_name_or_path=dataset_kwargs["dataset_name_or_path"],
        question_field=dataset_kwargs["question_column_name"],
        context_field=dataset_kwargs["context_column_name"],
        target_field=dataset_kwargs["answer_column_name"],
        tokenizer_name_or_path=embedding_name,
        train_batch_size=config.task_model_kwargs["train_batch_size"],
        eval_batch_size=config.task_model_kwargs["eval_batch_size"],
        tokenizer_kwargs=config.tokenizer_kwargs,
        batch_encoding_kwargs=config.batch_encoding_kwargs,
        load_dataset_kwargs={},
        dataloader_kwargs=config.dataloader_kwargs,
        doc_stride=config.task_model_kwargs["doc_stride"],
        use_cache=False,
        **config.datamodule_kwargs,
    )
    dm.setup("test")
    return dm


@pytest.fixture(scope="module")
def lightning_question_answering_pipeline(
    embedding_name: str,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    sample_question_answering_dataset: datasets.DatasetDict,
    dataset_kwargs: Dict[str, str],
) -> LIGHTNING_QA_PIPELINE_OUTPUT_TYPE:
    pipeline = LightningQuestionAnsweringPipeline(
        embedding_name_or_path=embedding_name,
        output_path=tmp_path_module,
        config=config,
        evaluation_filename="evaluation.json",
        model_checkpoint_kwargs={"filename": "last", "monitor": None, "save_last": False},
        **dataset_kwargs,
    )
    L.seed_everything(441, workers=True)
    metrics = pipeline.run()
    return pipeline, metrics


def test_lightning_qa_basic_config(basic_config: LightningQABasicConfig):
    assert isinstance(basic_config, LightningQABasicConfig)
    assert hasattr(basic_config, "finetune_last_n_layers")
    assert hasattr(basic_config, "task_train_kwargs")
    assert hasattr(basic_config, "task_model_kwargs")
    assert hasattr(basic_config, "datamodule_kwargs")
    assert hasattr(basic_config, "early_stopping_kwargs")
    assert hasattr(basic_config, "tokenizer_kwargs")
    assert hasattr(basic_config, "batch_encoding_kwargs")
    assert hasattr(basic_config, "dataloader_kwargs")
    assert hasattr(basic_config, "model_config_kwargs")
    assert isinstance(basic_config.task_model_kwargs, dict)
    assert "learning_rate" in basic_config.task_model_kwargs.keys()


def test_lightning_question_answering_pipeline(
    lightning_question_answering_pipeline: LIGHTNING_QA_PIPELINE_OUTPUT_TYPE,
):
    _, metrics = lightning_question_answering_pipeline
    metrics = metrics.metrics
    np.testing.assert_almost_equal(metrics["exact"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["f1"], 1.8518519, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["total"], 9.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["HasAns_f1"], 1.8518519, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["HasAns_total"], 9.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["best_f1"], 1.8518519, decimal=pytest.decimal)


def test_hf_model_exporter_from_pipeline(
    embedding_name: str,
    lightning_question_answering_pipeline: LIGHTNING_QA_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    comparison_metrics_keys: List[str],
    hf_datamodule: QuestionAnsweringDataModule,
):
    path = tmp_path_module / "hf_model_pipeline"
    exporter = HuggingFaceModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_question_answering_pipeline
    exporter.export_model_from_pipeline(pretrained_pipeline)

    loaded_model = AlbertForQuestionAnswering.from_pretrained(path)
    loaded_model_results = evaluate_hf_model_question_answering(
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
    lightning_question_answering_pipeline: LIGHTNING_QA_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    comparison_metrics_keys: List[str],
    hf_datamodule: QuestionAnsweringDataModule,
):
    path = tmp_path_module / "hf_model_task"
    exporter = HuggingFaceModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_question_answering_pipeline
    exporter.export_model_from_task(pretrained_pipeline.model.task)

    loaded_model = AlbertForQuestionAnswering.from_pretrained(path)
    loaded_model_results = evaluate_hf_model_question_answering(
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
    lightning_question_answering_pipeline: LIGHTNING_QA_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    comparison_metrics_keys: List[str],
    hf_datamodule: QuestionAnsweringDataModule,
):
    path = tmp_path_module / "onnx_model_pipeline"
    exporter = ONNXModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_question_answering_pipeline
    exporter.export_model_from_pipeline(pretrained_pipeline)

    loaded_model_results = evaluate_onnx_model_question_answering(
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
    lightning_question_answering_pipeline: LIGHTNING_QA_PIPELINE_OUTPUT_TYPE,
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    comparison_metrics_keys: List[str],
    hf_datamodule: QuestionAnsweringDataModule,
):
    path = tmp_path_module / "onnx_model_task"
    exporter = ONNXModelExporter(path=path)

    pretrained_pipeline, pretrained_metrics = lightning_question_answering_pipeline
    exporter.export_model_from_task(pretrained_pipeline.model.task)

    loaded_model_results = evaluate_onnx_model_question_answering(
        model_path=path, datamodule=hf_datamodule
    )
    assert_metrics_almost_equal(
        pretrained_model_metrics=pretrained_metrics,
        loaded_model_metrics=loaded_model_results,
        metric_keys=comparison_metrics_keys,
        decimal=pytest.decimal,
    )

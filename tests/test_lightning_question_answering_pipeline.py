from pathlib import Path

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lightning_config import LightningAdvancedConfig, LightningQABasicConfig
from embeddings.evaluator.evaluation_results import QuestionAnsweringEvaluationResults
from embeddings.pipeline.lightning_question_answering import LightningQuestionAnsweringPipeline
from tests.fixtures.sample_qa_dataset import sample_question_answering_dataset


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


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
        finetune_last_n_layers=0,
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
def lightning_question_answering_pipeline(
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
    sample_question_answering_dataset: datasets.DatasetDict,
):
    dataset = sample_question_answering_dataset
    dataset.save_to_disk(tmp_path_module / "data_sample")
    return LightningQuestionAnsweringPipeline(
        embedding_name_or_path="hf-internal-testing/tiny-albert",
        output_path=tmp_path_module,
        config=config,
        evaluation_filename="evaluation.json",
        predict_subset="validation",
        model_checkpoint_kwargs={"filename": "last", "monitor": None, "save_last": False},
        dataset_name_or_path=tmp_path_module / "data_sample",
    )


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
    lightning_question_answering_pipeline: LightningQuestionAnsweringPipeline,
):
    pl.seed_everything(441, workers=True)
    pipeline = lightning_question_answering_pipeline
    results = pipeline.run()
    assert isinstance(results, QuestionAnsweringEvaluationResults)

    metrics = results.metrics
    np.testing.assert_almost_equal(metrics["exact"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["f1"], 2.2222222, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["total"], 10.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["HasAns_f1"], 2.4691358, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["HasAns_total"], 9.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["best_f1"], 12.2222222, decimal=pytest.decimal)

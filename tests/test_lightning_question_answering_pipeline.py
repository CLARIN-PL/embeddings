from pathlib import Path

import datasets
import numpy as np
import pytest
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lightning_config import LightningQABasicConfig
from embeddings.evaluator.evaluation_results import QuestionAnsweringEvaluationResults
from embeddings.pipeline.lightning_question_answering import LightningQuestionAnsweringPipeline
from tests.fixtures.sample_qa_dataset import sample_question_answering_dataset


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def config() -> LightningQABasicConfig:
    return LightningQABasicConfig(
        finetune_last_n_layers=-1,
        task_train_kwargs={
            "max_epochs": 1,
            "devices": "auto",
            "accelerator": "auto",
            "deterministic": True,
        },
        task_model_kwargs={
            "learning_rate": 5e-4,
            "train_batch_size": 5,
            "eval_batch_size": 5,
            "use_scheduler": False,
            "optimizer": "Adam",
            "adam_epsilon": 1e-8,
            "warmup_steps": None,
            "weight_decay": 1e-3,
            "max_seq_length": 128,
            "doc_stride": 64,
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
        dataloader_kwargs={},
        model_config_kwargs={},
    )


@pytest.fixture(scope="module")
def lightning_question_answering_pipeline(
    config: LightningQABasicConfig,
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


def test_lightning_advanced_config(config: LightningQABasicConfig):
    lightning_config = config
    assert isinstance(lightning_config, LightningQABasicConfig)
    assert hasattr(lightning_config, "finetune_last_n_layers")
    assert hasattr(lightning_config, "task_train_kwargs")
    assert hasattr(lightning_config, "task_model_kwargs")
    assert hasattr(lightning_config, "datamodule_kwargs")
    assert hasattr(lightning_config, "early_stopping_kwargs")
    assert hasattr(lightning_config, "tokenizer_kwargs")
    assert hasattr(lightning_config, "batch_encoding_kwargs")
    assert hasattr(lightning_config, "dataloader_kwargs")
    assert hasattr(lightning_config, "model_config_kwargs")
    assert isinstance(lightning_config.task_model_kwargs, dict)
    assert "learning_rate" in lightning_config.task_model_kwargs.keys()


def test_lightning_question_answering_pipeline(
    lightning_question_answering_pipeline: LightningQuestionAnsweringPipeline,
):
    pipeline = lightning_question_answering_pipeline
    results = pipeline.run()
    assert isinstance(results, QuestionAnsweringEvaluationResults)

    metrics = results.metrics
    np.testing.assert_almost_equal(metrics["f1"], 10.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["total"], 10.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["HasAns_f1"], 0.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["HasAns_total"], 9.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["best_f1"], 10.0, decimal=pytest.decimal)
    np.testing.assert_almost_equal(metrics["exact"], 10.0, decimal=pytest.decimal)

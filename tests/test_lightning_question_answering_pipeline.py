from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.evaluator.evaluation_results import QuestionAnsweringEvaluationResults
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_question_answering import LightningQuestionAnsweringPipeline
from embeddings.task.lightning_task.question_answering import QuestionAnsweringTask


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def dataset_kwargs() -> Dict[str, Any]:
    return {
        "dataset_name_or_path": "./data_sample"
    }


@pytest.fixture(scope="module")
def config() -> LightningAdvancedConfig:
    return LightningAdvancedConfig(
        finetune_last_n_layers=0,
        task_train_kwargs={
            "max_epochs": 1,
            "devices": "auto",
            "accelerator": "auto",
        },
        task_model_kwargs={
            "learning_rate": 5e-4,
            "train_batch_size": 5,
            "eval_batch_size": 5,
            "use_scheduler": False,
            "optimizer": "AdamW",
            "adam_epsilon": 1e-8,
            "warmup_steps": None,
            "weight_decay": 1e-3,
            "max_seq_length": 512,
            "doc_stride": 128,
        },
        datamodule_kwargs={"max_seq_length": 64, },
        early_stopping_kwargs={"monitor": "val/Loss", "mode": "min", "patience": 3, },
        tokenizer_kwargs={},
        batch_encoding_kwargs={},
        dataloader_kwargs={},
        model_config_kwargs={},
    )


@pytest.fixture(scope="module")
def lightning_question_answering_pipeline():
    return LightningQuestionAnsweringPipeline(
        embedding_name_or_path="xlm-roberta-base",
        output_path=tmp_path_module,
        config=config,
        evaluation_filename="evaluation.json",
        predict_subset="dev",
        model_checkpoint_kwargs={
            "filename": "last",
            "monitor": None,
            "save_last": False
        },
        **dataset_kwargs,
    )


def test_lightning_advanced_config(
        config
):
    lightning_config = config
    assert isinstance(lightning_config, LightningAdvancedConfig)
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
        lightning_question_answering_pipeline: LightningQuestionAnsweringPipeline
):
    pipeline = lightning_question_answering_pipeline
    assert isinstance(pipeline, LightningQuestionAnsweringPipeline)

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl
from _pytest.tmpdir import TempdirFactory

from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def dataset_kwargs(tmp_path_module: Path) -> Dict[str, Any]:
    path = tmp_path_module
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/polemo2-official",
        load_dataset_kwargs={
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
        persist_path=str(path),
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": path,
        "input_column_name": ["text"],
        "target_column_name": "target",
    }


@pytest.fixture(scope="module")
def config() -> LightningAdvancedConfig:
    return LightningAdvancedConfig(
        finetune_last_n_layers=0,
        task_train_kwargs={
            "max_epochs": 3,
            "devices": "auto",
            "accelerator": "cpu",
            "deterministic": True,
        },
        task_model_kwargs={
            "learning_rate": 2e-4,
            "train_batch_size": 32,
            "eval_batch_size": 32,
            "use_scheduler": False,
            "optimizer": "AdamW",
            "adam_epsilon": 1e-8,
            "warmup_steps": 100,
            "weight_decay": 0.0,
        },
        datamodule_kwargs={"max_seq_length": None},
        model_config_kwargs={},
        tokenizer_kwargs={},
        batch_encoding_kwargs={},
        dataloader_kwargs={},
        early_stopping_kwargs={},
    )


@pytest.fixture(scope="module")
def lightning_classification_pipeline(
    dataset_kwargs: Dict[str, Any],
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
) -> LightningClassificationPipeline:
    result_path = tmp_path_module
    return LightningClassificationPipeline(
        embedding_name_or_path="allegro/herbert-base-cased",
        output_path=str(result_path),
        config=config,
        **dataset_kwargs,
    )


def test_lightning_classification_pipeline(
    lightning_classification_pipeline: LightningPipeline[
        datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]
    ],
) -> None:
    pl.seed_everything(441, workers=True)
    pipeline = lightning_classification_pipeline
    _ = pipeline.run()

    # val/Loss epoch 0: 1.3314
    # val/Loss epoch 1: 1.3253
    # val/Loss epoch 2: 1.3278

    np.testing.assert_equal(pipeline.model.task.best_epoch, 1)
    np.testing.assert_almost_equal(
        pipeline.model.task.best_validation_score, 1.325, decimal=pytest.decimal
    )

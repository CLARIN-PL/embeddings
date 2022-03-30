from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import pytest
import pytorch_lightning as pl

from embeddings.config.lightning_config import (
    LightningAdvancedConfig,
    LightningBasicConfig,
    LightningConfig,
)
from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline


@pytest.fixture(scope="module")
def dataset_kwargs() -> Tuple[Dict[str, Any], "TemporaryDirectory[str]"]:
    path = TemporaryDirectory()
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/polemo2-official",
        load_dataset_kwargs={
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
        persist_path=path.name,
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": path.name,
        "input_column_name": ["text"],
        "target_column_name": "target",
    }, path  # TemporaryDirectory object is passed additionally to omit cleanup of the temporal path


@pytest.fixture(scope="module")
def lightning_basic_config() -> LightningBasicConfig:
    return LightningBasicConfig(max_epochs=1, finetune_last_n_layers=0, max_seq_length=64)


@pytest.fixture(scope="module")
def lightning_advanced_config() -> LightningAdvancedConfig:
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
    )


@pytest.fixture(scope="module")
def lightning_config_missing_parameters() -> LightningAdvancedConfig:
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
            "use_scheduler": False,
        },
        datamodule_kwargs={
            "max_seq_length": 64,
        },
        early_stopping_kwargs={
            "monitor": "val/Loss",
            "mode": "min",
            "patience": 3,
        },
    )


def test_lightning_config_missing_params(
    lightning_config_missing_parameters: LightningConfig,
    dataset_kwargs: Dict[str, Any],
    result_path: "TemporaryDirectory[str]",
) -> None:
    pipeline = LightningClassificationPipeline(
        embedding_name_or_path="allegro/herbert-base-cased",
        output_path=result_path.name,
        config=lightning_config_missing_parameters,
        devices="auto",
        accelerator="cpu",
        **dataset_kwargs[0],
    )
    with pytest.raises(TypeError):
        pipeline.run()


def test_lightning_advanced_config_from_basic() -> None:
    basic_config = LightningBasicConfig()
    config = LightningAdvancedConfig.from_basic()
    for attr in config.__annotations__:
        assert getattr(config, attr) == getattr(basic_config, attr)


def test_lightning_classification_basic_config(
    lightning_basic_config: LightningConfig,
    dataset_kwargs: Dict[str, Any],
    result_path: "TemporaryDirectory[str]",
) -> None:
    pl.seed_everything(441, workers=True)
    pipeline = LightningClassificationPipeline(
        embedding_name_or_path="allegro/herbert-base-cased",
        output_path=result_path.name,
        config=lightning_basic_config,
        devices="auto",
        accelerator="cpu",
        **dataset_kwargs[0],
    )
    pipeline.run()


def test_lightning_classification_advanced_config(
    lightning_advanced_config: LightningConfig,
    dataset_kwargs: Dict[str, Any],
    result_path: "TemporaryDirectory[str]",
) -> None:
    pl.seed_everything(441, workers=True)
    pipeline = LightningClassificationPipeline(
        embedding_name_or_path="allegro/herbert-base-cased",
        output_path=result_path.name,
        config=lightning_advanced_config,
        devices="auto",
        accelerator="cpu",
        **dataset_kwargs[0],
    )
    pipeline.run()

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
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/polemo2-official",
        load_dataset_kwargs={
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
        persist_path=tmp_path_module.name,
        sample_missing_splits=None,
        ignore_test_subset=False,
        downsample_splits=(0.01, 0.01, 0.05),
        seed=441,
    )
    pipeline.run()

    return {
        "dataset_name_or_path": tmp_path_module.name,
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
def lightning_classification_pipeline(
    dataset_kwargs: Dict[str, Any],
    config: LightningAdvancedConfig,
    tmp_path_module: Path,
) -> LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]]:
    return LightningClassificationPipeline(
        embedding_name_or_path="allegro/herbert-base-cased",
        output_path=tmp_path_module.name,
        config=config,
        devices="auto",
        accelerator="cpu",
        **dataset_kwargs,
    )


def test_lightning_classification_pipeline(
    lightning_classification_pipeline: LightningPipeline[
        datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]
    ]
) -> None:
    pl.seed_everything(441, workers=True)
    pipeline = lightning_classification_pipeline
    result = pipeline.run()
    np.testing.assert_almost_equal(
        result["accuracy"]["accuracy"], 0.3783783, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["f1__average_macro"]["f1"], 0.1399999, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["precision__average_macro"]["precision"], 0.1, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["recall__average_macro"]["recall"], 0.2333333, decimal=pytest.decimal
    )

    assert "data" in result
    assert "y_pred" in result["data"]
    assert "y_true" in result["data"]
    assert "y_probabilities" in result["data"]
    assert "names" in result["data"]
    assert isinstance(result["data"]["y_pred"], np.ndarray)
    assert isinstance(result["data"]["y_true"], np.ndarray)
    assert isinstance(result["data"]["y_probabilities"], np.ndarray)
    assert isinstance(result["data"]["names"], np.ndarray)
    assert result["data"]["y_pred"].dtype == np.int64
    assert result["data"]["y_true"].dtype == np.int64
    assert result["data"]["y_probabilities"].dtype == np.float32
    assert isinstance(result["data"]["names"][0], str)

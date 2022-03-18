from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl
from _pytest.tmpdir import TempdirFactory

from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline


@pytest.fixture(scope="module")
def tmp_path_module(tmpdir_factory: TempdirFactory) -> Path:
    path = tmpdir_factory.mktemp(__name__)
    return Path(path)


@pytest.fixture(scope="module")
def pipeline_kwargs() -> Dict[str, Any]:
    return {"embedding_name_or_path": "allegro/herbert-base-cased"}


@pytest.fixture(scope="module")
def dataset_kwargs(tmp_path_module) -> Dict[str, Any]:
    path = str(tmp_path_module)
    pipeline = HuggingFacePreprocessingPipeline(
        dataset_name="clarin-pl/polemo2-official",
        load_dataset_kwargs={
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
        persist_path=path,
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
            "max_epochs": 1,
            "devices": "auto",
            "accelerator": "cpu",
            "deterministic": True,
        },
        task_model_kwargs={
            "learning_rate": 5e-4,
            "use_scheduler": False,
            "optimizer": "AdamW",
            "adam_epsilon": 1e-8,
            "warmup_steps": 100,
            "weight_decay": 0.0,
        },
        datamodule_kwargs={
            "downsample_train": 0.01,
            "downsample_val": 0.01,
            "downsample_test": 0.05,
        },
        dataloader_kwargs={"num_workers": 0},
    )


@pytest.fixture(scope="module")
def lightning_classification_pipeline(
    pipeline_kwargs: Dict[str, Any],
    dataset_kwargs: Dict[str, Any],
    config: LightningAdvancedConfig,
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]],
    "TemporaryDirectory[str]",
]:
    return (
        LightningClassificationPipeline(
            output_path=result_path.name,
            config=config,
            **pipeline_kwargs,
            **dataset_kwargs,
        ),
        result_path,
    )


def test_lightning_classification_pipeline(
    lightning_classification_pipeline: Tuple[
        LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]],
        "TemporaryDirectory[str]",
    ],
) -> None:
    pl.seed_everything(441, workers=True)
    pipeline, path = lightning_classification_pipeline
    result = pipeline.run()
    np.testing.assert_almost_equal(
        result["accuracy"]["accuracy"], 0.4864864, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["f1__average_macro"]["f1"], 0.2684458, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["precision__average_macro"]["precision"], 0.3602941, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["recall__average_macro"]["recall"], 0.325, decimal=pytest.decimal
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

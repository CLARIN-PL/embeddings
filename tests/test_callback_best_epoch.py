from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl

from embeddings.pipeline.hf_preprocessing_pipeline import HuggingFacePreprocessingPipeline
from embeddings.config.lightning_config import LightningAdvancedConfig
from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline


@pytest.fixture(scope="module")
def pipeline_kwargs() -> Dict[str, Any]:
    return {"embedding_name_or_path": "allegro/herbert-base-cased"}


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
def config() -> LightningAdvancedConfig:
    return LightningAdvancedConfig(
        finetune_last_n_layers=0,
        task_train_kwargs={
            "max_epochs": 3,
            "devices": "auto",
            "accelerator": "cpu",
            "deterministic": True,
        },
        task_model_kwargs={"learning_rate": 2e-4, "use_scheduler": False},
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
    _ = pipeline.run()

    # val/Loss epoch 0: 1.3314
    # val/Loss epoch 1: 1.3253
    # val/Loss epoch 2: 1.3278

    np.testing.assert_equal(pipeline.model.task.best_epoch, 1)
    np.testing.assert_almost_equal(
        pipeline.model.task.best_validation_score, 1.325, decimal=pytest.decimal
    )
    path.cleanup()

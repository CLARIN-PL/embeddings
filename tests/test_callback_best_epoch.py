from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl

from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline


@pytest.fixture(scope="module")
def pipeline_kwargs() -> Dict[str, Any]:
    return {"embedding_name_or_path": "allegro/herbert-base-cased", "finetune_last_n_layers": 0}


@pytest.fixture(scope="module")
def dataset_kwargs() -> Dict[str, Any]:
    return {
        "dataset_name_or_path": "clarin-pl/polemo2-official",
        "input_column_name": ["text"],
        "target_column_name": "target",
        "load_dataset_kwargs": {
            "train_domains": ["hotels", "medicine"],
            "dev_domains": ["hotels", "medicine"],
            "test_domains": ["hotels", "medicine"],
            "text_cfg": "text",
        },
    }


@pytest.fixture(scope="module")
def task_train_kwargs() -> Dict[str, Any]:
    return {
        "max_epochs": 3,
        "devices": "auto",
        "accelerator": "cpu",
        "deterministic": True,
    }


@pytest.fixture(scope="module")
def task_model_kwargs() -> Dict[str, Any]:
    return {"learning_rate": 2e-4, "use_scheduler": False}


@pytest.fixture(scope="module")
def datamodule_kwargs() -> Dict[str, Any]:
    return {
        "downsample_train": 0.01,
        "downsample_val": 0.01,
        "downsample_test": 0.05,
        "num_workers": 0,
    }


@pytest.fixture(scope="module")
def lightning_classification_pipeline(
    pipeline_kwargs: Dict[str, Any],
    dataset_kwargs: Dict[str, Any],
    datamodule_kwargs: Dict[str, Any],
    task_train_kwargs: Dict[str, Any],
    task_model_kwargs: Dict[str, Any],
    result_path: "TemporaryDirectory[str]",
) -> Tuple[
    LightningPipeline[datasets.DatasetDict, Dict[str, np.ndarray], Dict[str, Any]],
    "TemporaryDirectory[str]",
]:
    return (
        LightningClassificationPipeline(
            output_path=result_path.name,
            **pipeline_kwargs,
            **dataset_kwargs,
            datamodule_kwargs=datamodule_kwargs,
            task_train_kwargs=task_train_kwargs,
            task_model_kwargs=task_model_kwargs,
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

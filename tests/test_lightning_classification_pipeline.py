from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import datasets
import numpy as np
import pytest
import pytorch_lightning as pl

from embeddings.pipeline.lightning_classification import LightningClassificationPipeline
from embeddings.pipeline.lightning_pipeline import LightningPipeline


@pytest.fixture
def pipeline_kwargs() -> Dict[str, Any]:
    return {"embedding_name": "allegro/herbert-base-cased", "finetune_last_n_layers": 0}


@pytest.fixture
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


@pytest.fixture
def task_train_kwargs() -> Dict[str, Any]:
    return {
        "max_epochs": 1,
        "devices": "auto",
        "accelerator": "cpu",
        "deterministic": True,
    }


@pytest.fixture
def task_model_kwargs() -> Dict[str, Any]:
    return {"learning_rate": 5e-4, "use_scheduler": False}


@pytest.fixture
def datamodule_kwargs() -> Dict[str, Any]:
    return {
        "downsample_train": 0.01,
        "downsample_val": 0.01,
        "downsample_test": 0.05,
        "num_workers": 0,
    }


@pytest.fixture
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
    result = pipeline.run()
    path.cleanup()
    np.testing.assert_almost_equal(
        result["accuracy"]["accuracy"], 0.4864864864864865, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["f1__average_macro"]["f1"], 0.2684458398744113, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["precision__average_macro"]["precision"], 0.3602941176470588, decimal=pytest.decimal
    )
    np.testing.assert_almost_equal(
        result["recall__average_macro"]["recall"], 0.325, decimal=pytest.decimal
    )
